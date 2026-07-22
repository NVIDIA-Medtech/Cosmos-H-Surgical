# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from cosmos_h_surgical.checkpoints import (
    DEFAULT_MODEL_KEY,
    HF_REPOSITORY_ENV,
    HF_REVISION_ENV,
    MODEL_CONFIG_PATH,
    register_checkpoint_alias,
)
from cosmos_h_surgical.inference import (
    _RESIZE_MODE_EXTRA_KEY,
    _configure_distributed_timeout,
    _iter_transfer_safe_batches,
    _normalized_checkpoint_argv,
    _normalized_cli_argv,
    _with_default_checkpoint_argv,
    run_framework_cli,
)


def test_configure_distributed_timeout_updates_nccl_subgroup_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch.distributed import distributed_c10d

    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)
    monkeypatch.setattr(distributed_c10d, "default_pg_nccl_timeout", timedelta(seconds=600))

    _configure_distributed_timeout()

    assert os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "1800"
    assert distributed_c10d.default_pg_nccl_timeout == timedelta(seconds=1800)


def test_framework_cli_forwards_arguments_and_restores_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    original = list(sys.argv)
    observed: list[str] = []
    monkeypatch.delenv("COSMOS_TRAINING", raising=False)

    def fake_entrypoint() -> None:
        observed.extend(sys.argv)

    assert run_framework_cli(["-i", "sample.json", "--seed", "0"], entrypoint=fake_entrypoint) == 0
    assert observed == [
        "cosmos-h-surgical infer",
        "-i",
        "sample.json",
        "--seed",
        "0",
        "--checkpoint-path",
        "Cosmos-H-Surgical",
    ]
    assert sys.argv == original
    assert os.environ["COSMOS_TRAINING"] == "0"


def test_normalized_cli_argv_preserves_relative_path_context(tmp_path: Path) -> None:
    source = tmp_path / "transfer.json"
    source.write_text(
        json.dumps(
            {
                "name": "edge",
                "prompt_path": "../prompts/edge.json",
                "resize_mode": "stretch",
            }
        )
    )

    with _normalized_cli_argv(["-i", str(source), "--seed", "0"]) as (argv, needs_compat):
        normalized = Path(argv[1])
        assert needs_compat is True
        assert normalized != source
        assert normalized.parent == source.parent
        data = json.loads(normalized.read_text())
        assert "resize_mode" not in data
        assert data["extra"][_RESIZE_MODE_EXTRA_KEY] == "stretch"
        assert data["prompt_path"] == "../prompts/edge.json"

    assert not normalized.exists()
    assert json.loads(source.read_text())["resize_mode"] == "stretch"


def test_normalized_cli_argv_rejects_unknown_resize_mode(tmp_path: Path) -> None:
    source = tmp_path / "transfer.json"
    source.write_text(json.dumps({"resize_mode": "letterbox"}))

    with pytest.raises(ValueError, match="Unsupported resize_mode"):
        with _normalized_cli_argv(["-i", str(source)]):
            pass


def test_normalized_cli_argv_expands_quoted_glob_before_normalizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = tmp_path / "specs"
    specs.mkdir()
    first = specs / "a.json"
    second = specs / "b.json"
    first.write_text(json.dumps({"name": "a", "resize_mode": "stretch"}))
    second.write_text(json.dumps({"name": "b", "resize_mode": "stretch"}))
    monkeypatch.chdir(tmp_path)

    with _normalized_cli_argv(["-i", "specs/*.json", "specs/a.json", "--seed", "0"]) as (
        argv,
        needs_compat,
    ):
        input_paths = [Path(value) for value in argv[1:3]]
        assert argv[3:] == ["--seed", "0"]
        assert needs_compat is True
        assert [json.loads(path.read_text())["name"] for path in input_paths] == ["a", "b"]
        assert [json.loads(path.read_text())["name"] for path in sorted(input_paths)] == ["a", "b"]
        assert input_paths[0].resolve() != first.resolve()
        assert input_paths[0].parent.resolve() == specs.resolve()
        assert input_paths[1].resolve() != second.resolve()
        normalized_paths = input_paths

    assert not any(path.exists() for path in normalized_paths)
    assert first.exists()
    assert second.exists()


def test_normalized_cli_argv_rejects_unmatched_glob(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input pattern matched no files"):
        with _normalized_cli_argv(["-i", str(tmp_path / "*.json")]):
            pass


def test_transfer_safe_batches_serialize_transfer_samples_and_preserve_i2v_batches() -> None:
    class Sample:
        def __init__(self, name: str, transfer: bool = False) -> None:
            self.name = name
            self.transfer_hints = {"control": {}} if transfer else None

    samples = [
        Sample("i2v-a"),
        Sample("edge", transfer=True),
        Sample("blur", transfer=True),
        Sample("i2v-b"),
        Sample("i2v-c"),
    ]
    calls: list[list[str]] = []

    def original_create_batches(_inference: object, batch: list[Sample]):
        names = [sample.name for sample in batch]
        calls.append(names)
        yield names

    yielded = list(_iter_transfer_safe_batches(original_create_batches, object(), samples))

    assert calls == [["i2v-a"], ["edge"], ["blur"], ["i2v-b", "i2v-c"]]
    assert yielded == calls


def test_normalized_checkpoint_argv_adds_legacy_bias_field_without_copying_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "iter_000040000"
    checkpoint.mkdir()
    source_config = {"model": {"config": {"vision_gen": True}}}
    (checkpoint / "config.json").write_text(json.dumps(source_config))
    weight = checkpoint / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"weights")

    with _normalized_checkpoint_argv(["--checkpoint-path", str(checkpoint)]) as argv:
        migrated = Path(argv[1])
        assert migrated != checkpoint
        assert json.loads((migrated / "config.json").read_text())["model"]["config"]["enable_input_bias"] is True
        assert (migrated / weight.name).is_symlink()
        assert (migrated / weight.name).read_bytes() == b"weights"

    assert not migrated.exists()
    assert json.loads((checkpoint / "config.json").read_text()) == source_config


def test_normalized_checkpoint_argv_leaves_current_config_unchanged(tmp_path: Path) -> None:
    checkpoint = tmp_path / "current"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model": {"config": {"enable_input_bias": False}}}))

    original = ["--checkpoint-path", str(checkpoint)]
    with _normalized_checkpoint_argv(original) as argv:
        assert argv == original


@pytest.mark.parametrize(
    "argv",
    [
        ["-i", "sample.json", "--checkpoint-path", "/tmp/checkpoint"],
        ["-i", "sample.json", "--checkpoint-path=/tmp/checkpoint"],
    ],
)
def test_default_checkpoint_preserves_explicit_checkpoint(argv: list[str]) -> None:
    assert _with_default_checkpoint_argv(argv) == argv


def test_default_checkpoint_is_appended_when_omitted() -> None:
    assert _with_default_checkpoint_argv(["-i", "sample.json"]) == [
        "-i",
        "sample.json",
        "--checkpoint-path",
        "Cosmos-H-Surgical",
    ]


def test_checkpoint_alias_uses_staging_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference.args import _CHECKPOINTS

    previous = _CHECKPOINTS.get(DEFAULT_MODEL_KEY)
    monkeypatch.setenv(HF_REPOSITORY_ENV, "pengfeig/Cosmos-H-Surgical-staging")
    monkeypatch.setenv(HF_REVISION_ENV, "rc/v0.3.0-cosmos3")
    try:
        register_checkpoint_alias()
        checkpoint = _CHECKPOINTS[DEFAULT_MODEL_KEY]
        assert checkpoint.config_file == str(MODEL_CONFIG_PATH)
        assert checkpoint.hf.repository == "pengfeig/Cosmos-H-Surgical-staging"
        assert checkpoint.hf.revision == "rc/v0.3.0-cosmos3"
        assert checkpoint.hf.subdirectory == ""
    finally:
        if previous is None:
            _CHECKPOINTS.pop(DEFAULT_MODEL_KEY, None)
        else:
            _CHECKPOINTS[DEFAULT_MODEL_KEY] = previous
