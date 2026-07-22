# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta
from functools import wraps
from glob import glob
from pathlib import Path
from typing import Any

from cosmos_h_surgical.checkpoints import DEFAULT_MODEL_KEY, register_checkpoint_alias

_RESIZE_MODE_EXTRA_KEY = "_cosmos_h_surgical_resize_mode"
_RESIZE_MODE = ContextVar("cosmos_h_surgical_resize_mode", default="preserve_aspect")
_SUPPORTED_RESIZE_MODES = frozenset({"preserve_aspect", "stretch"})
_DISTRIBUTED_TIMEOUT_SECONDS = 1800


def _configure_distributed_timeout() -> None:
    timeout_value = os.environ.setdefault(
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        str(_DISTRIBUTED_TIMEOUT_SECONDS),
    )
    timeout_seconds = int(timeout_value)
    if timeout_seconds <= 0:
        raise ValueError("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC must be a positive integer")

    # DeviceMesh subgroups use PyTorch's NCCL default instead of the timeout
    # supplied when the framework initializes the default process group.
    from torch.distributed import distributed_c10d

    distributed_c10d.default_pg_nccl_timeout = timedelta(seconds=timeout_seconds)


def _normalize_resize_mode(data: dict[str, Any]) -> bool:
    mode = data.pop("resize_mode", None)
    if mode is None:
        return False
    if mode not in _SUPPORTED_RESIZE_MODES:
        choices = ", ".join(sorted(_SUPPORTED_RESIZE_MODES))
        raise ValueError(f"Unsupported resize_mode {mode!r}; expected one of: {choices}")

    extra = data.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("The inference field 'extra' must be an object when resize_mode is set")
    previous = extra.get(_RESIZE_MODE_EXTRA_KEY)
    if previous is not None and previous != mode:
        raise ValueError(f"Conflicting resize modes: {previous!r} and {mode!r}")
    extra[_RESIZE_MODE_EXTRA_KEY] = mode
    return True


def _normalize_input_file(path: Path) -> tuple[Path, bool]:
    if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
        return path, False

    if path.suffix == ".json":
        data: Any = json.loads(path.read_text())
        if not isinstance(data, dict) or not _normalize_resize_mode(data):
            return path, False
        serialized = json.dumps(data, indent=2) + "\n"
    else:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        changed = False
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON objects in {path}")
            changed = _normalize_resize_mode(record) or changed
        if not changed:
            return path, False
        serialized = "".join(json.dumps(record) + "\n" for record in records)

    # Keep the normalized file beside the source so relative paths retain the
    # exact semantics of Cosmos Framework's input loader.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".cosmos-h-surgical-compat-{path.name}-",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        return Path(temporary.name), True


def _expand_input_pattern(pattern: str) -> list[str]:
    if "*" not in pattern:
        return [pattern]

    matches = sorted(set(glob(pattern, recursive=True)))
    if not matches:
        raise FileNotFoundError(f"Input pattern matched no files: {pattern}")
    return matches


@contextmanager
def _normalized_cli_argv(argv: Sequence[str]):
    normalized = list(argv)
    temporary_paths: list[Path] = []
    needs_resize_compat = False
    index = 0
    try:
        while index < len(normalized):
            if normalized[index] not in {"-i", "--input-files"}:
                index += 1
                continue
            index += 1
            value_start = index
            while index < len(normalized) and not normalized[index].startswith("-"):
                index += 1
            value_end = index

            expanded = sorted(
                {
                    match
                    for pattern in normalized[value_start:value_end]
                    for match in _expand_input_pattern(pattern)
                }
            )
            normalized[value_start:value_end] = expanded
            index = value_start
            while index < value_start + len(expanded):
                replacement, changed = _normalize_input_file(Path(normalized[index]))
                if changed:
                    normalized[index] = str(replacement)
                    temporary_paths.append(replacement)
                    needs_resize_compat = True
                index += 1
        yield normalized, needs_resize_compat
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def _iter_transfer_safe_batches(
    original_create_batches: Callable[[Any, Sequence[Any]], Iterator[Any]],
    inference: Any,
    sample_args_list: Sequence[Any],
) -> Iterator[Any]:
    """Keep transfer collectives aligned while retaining regular I2V batching."""
    pending_non_transfer: list[Any] = []
    for sample_args in sample_args_list:
        if not bool(getattr(sample_args, "transfer_hints", None)):
            pending_non_transfer.append(sample_args)
            continue

        if pending_non_transfer:
            yield from original_create_batches(inference, pending_non_transfer)
            pending_non_transfer = []
        yield from original_create_batches(inference, [sample_args])

    if pending_non_transfer:
        yield from original_create_batches(inference, pending_non_transfer)


def _install_sequential_transfer_compat() -> None:
    from cosmos_framework.inference.inference import OmniInference

    if getattr(OmniInference, "_cosmos_h_surgical_sequential_transfer_compat", False):
        return

    original_create_batches = OmniInference.create_batches

    @wraps(original_create_batches)
    def create_batches(inference: Any, sample_args_list: Sequence[Any]) -> Iterator[Any]:
        yield from _iter_transfer_safe_batches(original_create_batches, inference, sample_args_list)

    OmniInference.create_batches = create_batches
    OmniInference._cosmos_h_surgical_sequential_transfer_compat = True


def _install_resize_mode_compat() -> None:
    import cosmos_framework.inference.transfer as transfer
    import cosmos_framework.inference.vision as vision

    if getattr(transfer, "_cosmos_h_surgical_resize_compat", False):
        return

    original_read_and_resize_media = vision.read_and_resize_media
    original_generate_transfer_sample = transfer.generate_transfer_sample

    @wraps(original_read_and_resize_media)
    def read_and_resize_media(path: Path, *, resolution: str, aspect_ratio: str | None, max_frames: int):
        mode = _RESIZE_MODE.get()
        if mode == "preserve_aspect":
            return original_read_and_resize_media(
                path,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                max_frames=max_frames,
            )

        raw_frames, fps = vision.read_media_frames(path, max_frames=max_frames)
        original_hw = (raw_frames.shape[2], raw_frames.shape[3])
        detected_aspect_ratio = vision.detect_aspect_ratio(raw_frames.shape[3], raw_frames.shape[2])
        final_aspect_ratio = aspect_ratio or detected_aspect_ratio
        width, height = vision.VIDEO_RES_SIZE_INFO[resolution][final_aspect_ratio]
        resized = vision.TF.resize(raw_frames.permute(1, 0, 2, 3), [height, width])
        return resized.permute(1, 0, 2, 3), fps, final_aspect_ratio, original_hw

    @wraps(original_generate_transfer_sample)
    def generate_transfer_sample(*, sample_args: Any, model: Any):
        mode = sample_args.extra.get(_RESIZE_MODE_EXTRA_KEY, "preserve_aspect")
        if mode not in _SUPPORTED_RESIZE_MODES:
            raise ValueError(f"Unsupported resize_mode in sample metadata: {mode!r}")
        token = _RESIZE_MODE.set(mode)
        try:
            return original_generate_transfer_sample(sample_args=sample_args, model=model)
        finally:
            _RESIZE_MODE.reset(token)

    vision.read_and_resize_media = read_and_resize_media
    transfer.read_and_resize_media = read_and_resize_media
    transfer.generate_transfer_sample = generate_transfer_sample
    transfer._cosmos_h_surgical_resize_compat = True


def _migrate_checkpoint_config(checkpoint_path: Path, destination: Path) -> bool:
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        return False

    config = json.loads(config_path.read_text())
    try:
        model_config = config["model"]["config"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unexpected Cosmos checkpoint config structure: {config_path}") from error
    if "enable_input_bias" in model_config:
        return False

    # Checkpoints produced before the public field was introduced always
    # instantiated these bias parameters, which is equivalent to True.
    model_config["enable_input_bias"] = True
    destination.mkdir()
    for source in checkpoint_path.iterdir():
        if source.name == "config.json":
            continue
        (destination / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return True


def _with_default_checkpoint_argv(argv: Sequence[str]) -> list[str]:
    normalized = list(argv)
    if any(
        argument == "--checkpoint-path" or argument.startswith("--checkpoint-path=")
        for argument in normalized
    ):
        return normalized
    return [*normalized, "--checkpoint-path", DEFAULT_MODEL_KEY]


@contextmanager
def _normalized_checkpoint_argv(argv: Sequence[str]):
    normalized = list(argv)
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    try:
        for index, argument in enumerate(normalized):
            if argument == "--checkpoint-path" and index + 1 < len(normalized):
                value_index = index + 1
            elif argument.startswith("--checkpoint-path="):
                value_index = index
            else:
                continue

            raw_value = normalized[value_index]
            checkpoint_value = raw_value.split("=", 1)[1] if value_index == index else raw_value
            checkpoint_path = Path(checkpoint_value).expanduser().absolute()
            if not checkpoint_path.is_dir():
                break

            temporary_directory = tempfile.TemporaryDirectory(
                prefix=".cosmos-h-surgical-checkpoint-compat-",
                dir=checkpoint_path.parent,
            )
            migrated_path = Path(temporary_directory.name) / "checkpoint"
            if not _migrate_checkpoint_config(checkpoint_path, migrated_path):
                temporary_directory.cleanup()
                temporary_directory = None
                break
            normalized[value_index] = (
                f"--checkpoint-path={migrated_path}" if value_index == index else str(migrated_path)
            )
            break
        yield normalized
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def run_framework_cli(
    argv: Sequence[str],
    *,
    entrypoint: Callable[[], Any] | None = None,
) -> int:
    """Run the pinned Cosmos Framework inference CLI with surgical compatibility."""
    os.environ["COSMOS_TRAINING"] = "0"
    _configure_distributed_timeout()
    with _normalized_cli_argv(argv) as (input_argv, needs_resize_compat):
        with _normalized_checkpoint_argv(_with_default_checkpoint_argv(input_argv)) as normalized_argv:
            if needs_resize_compat:
                _install_resize_mode_compat()
            if entrypoint is None:
                register_checkpoint_alias()
                _install_sequential_transfer_compat()
                from cosmos_framework.scripts.inference import main as entrypoint

            original_argv = sys.argv
            sys.argv = ["cosmos-h-surgical infer", *normalized_argv]
            try:
                result = entrypoint()
            finally:
                sys.argv = original_argv
    return 0 if result is None else int(result)
