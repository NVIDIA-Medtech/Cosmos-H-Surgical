# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos-H surgical JSON video dataset for Cosmos 3 SFT training."""

import argparse
import json
import math
import os
import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cosmos_framework.data.generator.local_datasets.helper import ffmpeg_decode_video, get_video_metadata
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
from cosmos_framework.inference.structured_caption import CAPTION_JSON_KEY, caption_json_to_prompt
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

_MAX_NUM_TOKENS = 2048  # Aligned with Nano SFT experiment config
_DATA_LOADER_SEED = int(os.environ.get("DATA_LOADER_SEED", 0))
_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
_SHELL_QUOTE_CHARS = "\"'\u2018\u2019\u201c\u201d"


def _split_comma_separated(value: str) -> list[str]:
    items = []
    for raw_item in value.split(","):
        item = raw_item.strip().strip(_SHELL_QUOTE_CHARS).strip()
        if item:
            items.append(item)
    return items


def _as_list(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return _split_comma_separated(value)
    return value


def _parse_factors(value: str | list[float]) -> list[float]:
    if isinstance(value, str):
        try:
            return [float(item) for item in _split_comma_separated(value)]
        except ValueError as error:
            raise ValueError(
                f"Invalid enlarged_factor={value!r}; expected comma-separated numbers such as '1.0,0.5'"
            ) from error
    return [float(item) for item in value]


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _first_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for child in value.values():
            result = _first_string(child)
            if result:
                return result
    if isinstance(value, list):
        for child in value:
            result = _first_string(child)
            if result:
                return result
    return ""


def _flatten_caption_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, str):
        return [(prefix, value)]
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, str | dict):
                items.append((child_prefix, child))
            if isinstance(child, dict | list):
                items.extend(_flatten_caption_values(child, child_prefix))
        return items
    if isinstance(value, list):
        items: list[tuple[str, Any]] = []
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            if isinstance(child, str | dict):
                items.append((child_prefix, child))
            if isinstance(child, dict | list):
                items.extend(_flatten_caption_values(child, child_prefix))
        return items
    return []


def _caption_value_to_text(caption_key: str, raw: Any) -> tuple[str, bool]:
    if isinstance(raw, dict):
        return caption_json_to_prompt(raw), True
    caption_text = str(raw).strip()
    is_structured = caption_key == CAPTION_JSON_KEY or caption_key.endswith(f".{CAPTION_JSON_KEY}")
    return caption_text, is_structured


class SurgicalVideoJSONDataset(torch.utils.data.IterableDataset):
    """Iterable Cosmos 3 port of Cosmos Predict 2.5 ``VideoDatasetJSON``.

    The manifest format is the same as the 2.5 dataset:

    .. code-block:: json

        {
          "training": [
            {"video": "relative/or/absolute/path.mp4"}
          ]
        }

    Captions are loaded from a sidecar file next to each video, with the same
    basename and either ``.json`` or ``.txt`` depending on ``caption_format``.
    For JSON captions, ``caption_types_and_weights`` can be used to randomly
    sample among available caption styles, e.g. ``{"short": 0.8, "long": 0.2}``.
    """

    def __init__(
        self,
        dataset_dir: str | list[str],
        json_path: str | list[str],
        enlarged_factor: str | list[float],
        num_frames: int,
        video_size: tuple[int, int],
        caption_format: str = "json",
        prompt_type: str | None = "short",
        caption_types_and_weights: dict[str, float] | None = None,
        tokenizer_config: Any | None = None,
        cfg_dropout_rate: float = 0.0,
        use_system_prompt: bool = False,
        append_duration_fps_timestamps: bool = True,
        append_resolution_info: bool = True,
        cfg_dropout_keep_metadata: bool = False,
        caption_suffix: str = "",
        conditioning_fps: float = -1,
        conditioning_fps_noise_std: float = 0.0,
        conditioning_config: dict[int, float] | None = None,
        temporal_compression_factor: int = 4,
        frame_selection_mode: str = "random",
        num_decode_threads: int = 2,
    ) -> None:
        super().__init__()
        assert caption_format in ("json", "text"), f"Unknown caption_format={caption_format!r}"
        assert frame_selection_mode in ("first", "center", "random"), (
            f"Unknown frame_selection_mode={frame_selection_mode!r}"
        )
        assert num_frames > 0, "num_frames must be positive"
        assert temporal_compression_factor >= 1, "temporal_compression_factor must be >= 1"
        assert tokenizer_config is not None, "tokenizer_config is required for Cosmos 3 sequence packing"

        self.dataset_dir = _as_list(dataset_dir)
        self.json_path = _as_list(json_path)
        self.enlarged_factor = _parse_factors(enlarged_factor)
        assert len(self.dataset_dir) == len(self.json_path) == len(self.enlarged_factor), (
            "dataset_dir, json_path, and enlarged_factor must have the same length"
        )

        self.num_frames = num_frames
        self.video_size = video_size
        self.caption_format = caption_format
        self.prompt_type = prompt_type
        self.caption_types_and_weights = self._normalize_caption_types_and_weights(caption_types_and_weights)
        self.cfg_dropout_rate = cfg_dropout_rate
        self.use_system_prompt = use_system_prompt
        self.append_duration_fps_timestamps = append_duration_fps_timestamps
        self.append_resolution_info = append_resolution_info
        self.cfg_dropout_keep_metadata = cfg_dropout_keep_metadata
        self.caption_suffix = caption_suffix.strip()
        self.conditioning_fps = conditioning_fps
        self.conditioning_fps_noise_std = conditioning_fps_noise_std
        self.temporal_compression_factor = temporal_compression_factor
        self.frame_selection_mode = frame_selection_mode
        self.num_decode_threads = num_decode_threads

        self.conditioning_config: dict[int, float] | None = None
        if conditioning_config is not None:
            normalized_config = {int(k): float(v) for k, v in conditioning_config.items()}
            total_prob = sum(normalized_config.values())
            assert total_prob > 0, "conditioning_config probabilities must sum to a positive number"
            self.conditioning_config = {k: v / total_prob for k, v in normalized_config.items()}
            log.info(f"Conditioning config: {self.conditioning_config}")

        self.video_paths = self._load_video_paths()
        self.num_failed_loads = 0

        # These are set by RankPartitionedDataLoader.
        self.shard_world_size = None
        self.shard_rank = None
        self.shard_id = 0
        self.is_initialized = False

        vlm_processor_or_tokenizer = lazy_instantiate(tokenizer_config)
        self.vlm_tokenizer = getattr(vlm_processor_or_tokenizer, "tokenizer", vlm_processor_or_tokenizer)
        self.vlm_tokenizer, _ = add_special_tokens(self.vlm_tokenizer)

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video_paths(self) -> list[str]:
        all_video_paths: list[str] = []
        for dataset_dir, json_path, enlarged_factor in zip(self.dataset_dir, self.json_path, self.enlarged_factor):
            manifest = _load_json(json_path)
            if isinstance(manifest, dict):
                if "training" not in manifest:
                    raise ValueError(f"Expected key 'training' in {json_path}; found {list(manifest.keys())}")
                entries = manifest["training"]
            elif isinstance(manifest, list):
                entries = manifest
            else:
                raise ValueError(f"Expected {json_path} to contain a dict or list, got {type(manifest)}")

            video_paths = []
            for entry in entries:
                video_relpath = entry["video"] if isinstance(entry, dict) else entry
                video_paths.append(os.path.join(dataset_dir, video_relpath))
            video_paths = sorted(video_paths)
            original_length = len(video_paths)

            rng = random.Random(_DATA_LOADER_SEED)
            rng.shuffle(video_paths)
            if enlarged_factor >= 1:
                video_paths = (video_paths * math.ceil(enlarged_factor))[: int(original_length * enlarged_factor)]
            else:
                video_paths = video_paths[: int(original_length * enlarged_factor)]

            log.warning(
                f"json_path: {json_path} enlarged factor is {enlarged_factor}, "
                f"original length is {original_length}, new length is {len(video_paths)}"
            )
            all_video_paths.extend(video_paths)

        all_video_paths = sorted(all_video_paths)
        random.Random(_DATA_LOADER_SEED).shuffle(all_video_paths)
        log.warning(f"{len(all_video_paths)} videos in total")
        return all_video_paths

    def _normalize_caption_types_and_weights(
        self, caption_types_and_weights: dict[str, float] | None
    ) -> dict[str, float] | None:
        if caption_types_and_weights is None:
            return None
        normalized_config = {str(k): float(v) for k, v in caption_types_and_weights.items()}
        total_prob = sum(v for v in normalized_config.values() if v > 0)
        assert total_prob > 0, "caption_types_and_weights probabilities must sum to a positive number"
        normalized_config = {k: v / total_prob for k, v in normalized_config.items() if v > 0}
        log.info(f"Caption types and weights: {normalized_config}")
        return normalized_config

    def _tokenize_caption(self, caption: str) -> tuple[list[int], str]:
        text_ids = tokenize_caption(
            caption,
            self.vlm_tokenizer,
            is_video=True,
            use_system_prompt=self.use_system_prompt,
        )
        if len(text_ids) > _MAX_NUM_TOKENS:
            log.warning(f"Text ids are too long, truncating: {len(text_ids)} > {_MAX_NUM_TOKENS}")
        text_ids = text_ids[:_MAX_NUM_TOKENS]
        return text_ids, caption

    def _select_weighted_json_caption(
        self, captions: Any, caption_types_and_weights: dict[str, float], rng: random.Random
    ) -> tuple[str, str, bool] | None:
        flat_captions = _flatten_caption_values(captions)
        candidates: list[tuple[str, str, bool]] = []
        weights: list[float] = []
        for caption_type, weight in caption_types_and_weights.items():
            for caption_key, raw_caption in flat_captions:
                if caption_key == caption_type or caption_key.endswith(f".{caption_type}"):
                    caption_text, used_structured_caption = _caption_value_to_text(caption_key, raw_caption)
                    if caption_text:
                        candidates.append((caption_text, caption_key, used_structured_caption))
                        weights.append(weight)
        if not candidates:
            return None
        return rng.choices(candidates, weights=weights, k=1)[0]

    def _load_json_caption(self, caption_path: Path, rng: random.Random) -> tuple[str, str, bool]:
        try:
            content = caption_path.read_text()
            captions = json.loads(content if content.strip().startswith("{") else "{" + content + "}")
            if self.caption_types_and_weights is not None:
                selected = self._select_weighted_json_caption(captions, self.caption_types_and_weights, rng)
                if selected is not None:
                    return selected
                log.warning(
                    f"No weighted caption type found in {caption_path}. "
                    f"Requested: {list(self.caption_types_and_weights.keys())}. Using prompt_type fallback."
                )
            if self.prompt_type and self.prompt_type in captions:
                caption_text, used_structured_caption = _caption_value_to_text(
                    self.prompt_type, captions[self.prompt_type]
                )
                return caption_text, self.prompt_type, used_structured_caption
            if self.prompt_type:
                for model_key, model_captions in captions.items():
                    if isinstance(model_captions, dict) and self.prompt_type in model_captions:
                        caption_key = f"{model_key}.{self.prompt_type}"
                        caption_text, used_structured_caption = _caption_value_to_text(
                            caption_key, model_captions[self.prompt_type]
                        )
                        return caption_text, caption_key, used_structured_caption
                log.warning(
                    f"Prompt type {self.prompt_type!r} not found in {caption_path}. "
                    f"Available top-level keys: {list(captions.keys())}. Using first available."
                )
            if CAPTION_JSON_KEY in captions:
                caption_text, used_structured_caption = _caption_value_to_text(
                    CAPTION_JSON_KEY, captions[CAPTION_JSON_KEY]
                )
                return caption_text, CAPTION_JSON_KEY, used_structured_caption
            return _first_string(captions), "first_available", False
        except Exception as e:
            log.warning(f"Failed to read JSON caption file {caption_path}: {e}")
            return "", "missing", False

    def _load_text_caption(self, caption_path: Path) -> tuple[str, str, bool]:
        try:
            return caption_path.read_text().strip(), "text", False
        except Exception as e:
            log.warning(f"Failed to read caption file {caption_path}: {e}")
            return "", "missing", False

    def _load_caption(self, video_path: str, rng: random.Random) -> tuple[str, str, bool]:
        caption_path = Path(video_path).with_suffix(".json" if self.caption_format == "json" else ".txt")
        if self.caption_format == "json":
            return self._load_json_caption(caption_path, rng)
        return self._load_text_caption(caption_path)

    def _select_frame_range(self, total_frames: int, rng: random.Random) -> tuple[int, int]:
        if total_frames < self.num_frames:
            raise ValueError(f"Video has only {total_frames} frames, at least {self.num_frames} frames are required.")

        max_start_idx = total_frames - self.num_frames
        if self.frame_selection_mode == "first" or max_start_idx <= 0:
            start_frame = 0
        elif self.frame_selection_mode == "center":
            start_frame = max_start_idx // 2
        elif self.frame_selection_mode == "random":
            start_frame = rng.randint(0, max_start_idx)
        else:
            raise ValueError(f"Unknown frame_selection_mode: {self.frame_selection_mode}")
        end_frame = start_frame + self.num_frames - 1
        return start_frame, end_frame

    def _decode_video(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        target_h, target_w = self.video_size
        video_chunk = []
        for idx, frame in enumerate(
            ffmpeg_decode_video(video_path, scale_hw=(target_h, target_w), num_threads=self.num_decode_threads)
        ):
            if idx < start_frame:
                continue
            if idx <= end_frame:
                video_chunk.append(frame)
            else:
                break

        if len(video_chunk) != self.num_frames:
            raise ValueError(
                f"Decoded {len(video_chunk)} frames from {video_path}, expected {self.num_frames} "
                f"(start={start_frame}, end={end_frame})"
            )

        video_np = np.stack(video_chunk, axis=0)  # [T,H,W,3]
        target_t = (video_np.shape[0] - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1
        video_np = video_np[:target_t]
        video_np = np.transpose(video_np, (3, 0, 1, 2))  # [3,T,H,W]
        return torch.from_numpy(np.ascontiguousarray(video_np)).to(torch.uint8)

    def process_one_sample(self, video_path: str, rng: random.Random) -> dict | None:
        try:
            video_info = get_video_metadata(video_path)
            original_fps = video_info["fps"]
            total_frames = video_info["total_frames"]
            start_frame, end_frame = self._select_frame_range(total_frames, rng)
            video = self._decode_video(video_path, start_frame, end_frame)

            caption, caption_key, used_structured_caption = self._load_caption(video_path, rng)
            caption = caption.strip()
            if caption and not used_structured_caption:
                caption = caption.rstrip(".") + "."

            cond_fps = original_fps if self.conditioning_fps < 0 else self.conditioning_fps
            if self.conditioning_fps_noise_std > 0:
                cond_fps *= math.exp(rng.gauss(0.0, self.conditioning_fps_noise_std))

            if self.caption_suffix:
                caption = (caption + " " + self.caption_suffix).strip()

            if self.cfg_dropout_keep_metadata and self.cfg_dropout_rate > 0 and rng.random() < self.cfg_dropout_rate:
                caption = ""

            target_h, target_w = self.video_size
            if self.append_duration_fps_timestamps:
                duration = video.shape[1] / cond_fps
                caption = caption + " " + _DURATION_TEMPLATE.format(duration=duration, fps=cond_fps)
            if self.append_resolution_info:
                caption = caption + " " + _RESOLUTION_TEMPLATE.format(height=target_h, width=target_w)
            caption = caption.strip()

            if (
                not self.cfg_dropout_keep_metadata
                and self.cfg_dropout_rate > 0
                and rng.random() < self.cfg_dropout_rate
            ):
                caption = ""

            text_ids, caption = self._tokenize_caption(caption)
            sample = dict(
                __key__=Path(video_path).stem,
                __url__=video_path,
                fps=original_fps,
                n_orig_video_frames=total_frames,
                frame_start=start_frame,
                frame_end=end_frame,
                num_frames=video.shape[1],
                video=video,
                num_multiplier=1,
                conditioning_fps=cond_fps,
                padding_mask=torch.zeros((1, target_h, target_w), dtype=torch.float32),
                image_size=torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float32),
                ai_caption=caption,
                sampled_caption_style=caption_key,
                text_token_ids=torch.tensor(text_ids),
            )

            if self.conditioning_config is not None:
                t_latent = 1 + (video.shape[1] - 1) // self.temporal_compression_factor
                frames_options = list(self.conditioning_config.keys())
                weights = list(self.conditioning_config.values())
                num_cond = rng.choices(frames_options, weights=weights, k=1)[0]
                num_cond = max(0, min(num_cond, t_latent - 1))
                sample["sequence_plan"] = SequencePlan(
                    has_text=True,
                    has_vision=True,
                    condition_frame_indexes_vision=list(range(num_cond)),
                )

            return sample
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {video_path} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            return None

    def __iter__(self):
        assert not self.is_initialized, "Dataset can only be initialized once."
        assert len(self.video_paths) > 0, "Did not find any data."

        if self.shard_world_size is not None:
            train_world_size = self.shard_world_size
            train_rank = self.shard_rank
            log.info(f"Using shard_world_size: {train_world_size} and shard_rank: {train_rank}", rank0_only=False)
        else:
            train_world_size = torch.distributed.get_world_size()
            train_rank = torch.distributed.get_rank()
        train_dp_rank = train_rank
        train_num_dp_groups = train_world_size
        train_dp_group_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            total_data_ranks = worker_info.num_workers * train_world_size
            data_rank = worker_info.id + train_rank * worker_info.num_workers
            seed = worker_info.seed
        else:
            log.warning("No data worker info found. Using default worker rank and number of workers.", rank0_only=False)
            total_data_ranks = train_world_size
            data_rank = train_rank
            seed = _DATA_LOADER_SEED

        log.info(
            f"train_world_size: {train_world_size}; "
            f"train_rank: {train_rank}; "
            f"train_dp_rank: {train_dp_rank}; "
            f"train_num_dp_groups: {train_num_dp_groups}; "
            f"train_dp_group_size: {train_dp_group_size}; "
            f"worker_info: {worker_info}; "
            f"total_data_ranks: {total_data_ranks}; "
            f"data_rank: {data_rank}; "
            f"data_loader_seed: {_DATA_LOADER_SEED}; "
            f"seed: {seed}; "
            f"shard_id: {self.shard_id}; "
            f"shard_world_size: {self.shard_world_size}; "
            f"shard_rank: {self.shard_rank}",
            rank0_only=False,
        )

        video_paths = list(self.video_paths)
        multiplier = max(1, total_data_ranks * 50 // len(video_paths))
        video_paths = video_paths * multiplier
        remainder = len(video_paths) % total_data_ranks
        if remainder:
            video_paths.extend(video_paths[: total_data_ranks - remainder])

        random.Random(_DATA_LOADER_SEED + self.shard_id).shuffle(video_paths)
        video_paths = video_paths[data_rank::total_data_ranks]
        log.info(
            f"DRank {data_rank}/{total_data_ranks} has {len(video_paths)} videos "
            f"from {len(set(video_paths))} unique paths.",
            rank0_only=False,
        )

        self.is_initialized = True
        rng = random.Random(_DATA_LOADER_SEED + seed + data_rank + self.shard_id * 12345)
        while True:
            rng.shuffle(video_paths)
            for video_path in video_paths:
                sample = self.process_one_sample(video_path, rng)
                if sample is not None:
                    yield sample


def get_surgical_video_json_dataset(
    dataset_dir: str | list[str],
    json_path: str | list[str],
    enlarged_factor: str | list[float],
    num_frames: int = 93,
    video_size: tuple[int, int] = (704, 1280),
    caption_format: str = "json",
    prompt_type: str | None = "short",
    caption_types_and_weights: dict[str, float] | None = None,
    tokenizer_config: Any | None = None,
    cfg_dropout_rate: float = 0.1,
    use_system_prompt: bool = False,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    cfg_dropout_keep_metadata: bool = False,
    caption_suffix: str = "",
    conditioning_fps: float = -1,
    conditioning_fps_noise_std: float = 0.0,
    conditioning_config: dict[int, float] | None = None,
    temporal_compression_factor: int = 4,
    frame_selection_mode: str = "random",
    num_decode_threads: int = 2,
    **kwargs,
) -> SurgicalVideoJSONDataset:
    """Create a Cosmos-H surgical JSON dataset for Cosmos 3 training."""
    if kwargs:
        log.info(f"Unknown kwargs for get_surgical_video_json_dataset: {kwargs}")
    return SurgicalVideoJSONDataset(
        dataset_dir=dataset_dir,
        json_path=json_path,
        enlarged_factor=enlarged_factor,
        num_frames=num_frames,
        video_size=video_size,
        caption_format=caption_format,
        prompt_type=prompt_type,
        caption_types_and_weights=caption_types_and_weights,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        use_system_prompt=use_system_prompt,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        cfg_dropout_keep_metadata=cfg_dropout_keep_metadata,
        caption_suffix=caption_suffix,
        conditioning_fps=conditioning_fps,
        conditioning_fps_noise_std=conditioning_fps_noise_std,
        conditioning_config=conditioning_config,
        temporal_compression_factor=temporal_compression_factor,
        frame_selection_mode=frame_selection_mode,
        num_decode_threads=num_decode_threads,
    )


def _parse_weight_mapping(value: str, key_type: type = str) -> dict | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = {}
        for item in value.split(","):
            if not item.strip():
                continue
            key, weight = item.split("=", 1)
            parsed[key.strip()] = float(weight)
    return {key_type(k): float(v) for k, v in parsed.items()}


def _default_tokenizer_config(config_variant: str, pretrained_model_name: str) -> dict[str, str]:
    return {
        "_target_": "cosmos_framework.configs.base.defaults.vlm.create_qwen2_tokenizer_with_download",
        "config_variant": config_variant,
        "pretrained_model_name": pretrained_model_name,
    }


def _run_manifest_only(args: argparse.Namespace) -> None:
    dataset_dirs = _as_list(args.dataset_dir)
    json_paths = _as_list(args.json_path)
    for dataset_dir, json_path in zip(dataset_dirs, json_paths):
        manifest = _load_json(json_path)
        entries = manifest["training"] if isinstance(manifest, dict) and "training" in manifest else manifest
        entry = entries[0]
        video_relpath = entry["video"] if isinstance(entry, dict) else entry
        video_path = os.path.join(dataset_dir, video_relpath)
        caption_path = Path(video_path).with_suffix(".json" if args.caption_format == "json" else ".txt")
        print(f"manifest: {json_path}")
        print(f"  entries: {len(entries)}")
        print(f"  first_video: {video_path}")
        print(f"  video_exists: {Path(video_path).exists()}")
        print(f"  caption_path: {caption_path}")
        print(f"  caption_exists: {caption_path.exists()}")
        if args.caption_format == "json" and caption_path.exists():
            captions = _load_json(str(caption_path))
            print(f"  caption_keys: {list(captions)[:10] if isinstance(captions, dict) else type(captions)}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test SurgicalVideoJSONDataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--enlarged-factor", default="1.0")
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--caption-format", choices=("json", "text"), default="json")
    parser.add_argument("--prompt-type", default="short")
    parser.add_argument(
        "--caption-types-and-weights",
        default="",
        help='Optional weighted caption sampler, e.g. "short=0.8,long=0.2" or JSON {"short": 0.8, "long": 0.2}.',
    )
    parser.add_argument(
        "--conditioning-config",
        default="1=1.0",
        help='Optional I2W/V2W conditioning distribution, e.g. "1=1.0" or JSON {"1": 1.0}. Empty disables.',
    )
    parser.add_argument("--frame-selection-mode", choices=("first", "center", "random"), default="random")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer-config-variant",
        choices=("hf", "gcp", "s3"),
        default="hf",
        help="Tokenizer source for the standalone smoke test. Use hf to avoid Cosmos easy_io object-store setup.",
    )
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--manifest-only", action="store_true", help="Only verify manifest/video/caption paths.")
    args = parser.parse_args()

    if args.manifest_only:
        _run_manifest_only(args)
        return

    prompt_type = None if args.prompt_type.lower() == "none" else args.prompt_type
    dataset = SurgicalVideoJSONDataset(
        dataset_dir=args.dataset_dir,
        json_path=args.json_path,
        enlarged_factor=args.enlarged_factor,
        num_frames=args.num_frames,
        video_size=(args.height, args.width),
        caption_format=args.caption_format,
        prompt_type=prompt_type,
        caption_types_and_weights=_parse_weight_mapping(args.caption_types_and_weights, str),
        tokenizer_config=_default_tokenizer_config(args.tokenizer_config_variant, args.tokenizer_model),
        cfg_dropout_rate=0.0,
        conditioning_config=_parse_weight_mapping(args.conditioning_config, int),
        frame_selection_mode=args.frame_selection_mode,
    )

    rng = random.Random(args.seed)
    video_path = dataset.video_paths[args.index % len(dataset.video_paths)]
    sample = dataset.process_one_sample(video_path, rng)
    if sample is None:
        raise RuntimeError(f"Failed to load sample from {video_path}")

    print(f"num_videos: {len(dataset)}")
    print(f"selected_video: {video_path}")
    print(f"sample_key: {sample['__key__']}")
    print(f"video: shape={tuple(sample['video'].shape)}, dtype={sample['video'].dtype}")
    print(f"text_token_ids: shape={tuple(sample['text_token_ids'].shape)}, dtype={sample['text_token_ids'].dtype}")
    print(f"caption_style: {sample['sampled_caption_style']}")
    print(f"caption_preview: {sample['ai_caption'][:300]}")
    print(f"fps: {sample['fps']}, conditioning_fps: {sample['conditioning_fps']}")
    print(f"frame_start: {sample['frame_start']}, frame_end: {sample['frame_end']}")
    print(f"image_size: {sample['image_size'].tolist()}")
    print(f"padding_mask: shape={tuple(sample['padding_mask'].shape)}, dtype={sample['padding_mask'].dtype}")
    if "sequence_plan" in sample:
        print(f"sequence_plan: {sample['sequence_plan'].as_dict()}")


if __name__ == "__main__":
    _main()
