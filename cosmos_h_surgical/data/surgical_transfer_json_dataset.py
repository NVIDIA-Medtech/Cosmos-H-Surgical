# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local surgical video-transfer dataset for Cosmos 3 mixed-capability SFT.

The dataset shares manifest/caption/sharding behavior with
``SurgicalVideoJSONDataset`` and adds ControlNet-style transfer samples where a
sidecar or derived control video is fully conditioned and the aligned RGB video
is generated.
"""

from __future__ import annotations

import math
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cosmos_framework.data.generator.augmentors.transfer_control_input.control_input import (
    AddControlInputBlur,
    AddControlInputDepth,
    AddControlInputEdge,
    AddControlInputSeg,
)
from cosmos_framework.data.generator.local_datasets.helper import get_video_metadata
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.utils import log

from cosmos_h_surgical.data.surgical_video_json_dataset import (
    _DURATION_TEMPLATE,
    _RESOLUTION_TEMPLATE,
    SurgicalVideoJSONDataset,
)

_SUPPORTED_MODALITIES = {"edge", "blur", "depth", "seg"}


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        weights = {"edge": 1.0, "blur": 1.0, "depth": 1.0, "seg": 1.0}
    normalized = {str(k): float(v) for k, v in weights.items() if float(v) > 0}
    unknown = set(normalized) - _SUPPORTED_MODALITIES
    if unknown:
        raise ValueError(
            f"Unsupported transfer modalities {sorted(unknown)}; supported: {sorted(_SUPPORTED_MODALITIES)}"
        )
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("control_modalities probabilities must sum to a positive number")
    return {key: value / total for key, value in normalized.items()}


def _resolve_sidecar_path(video_path: str, suffix: str) -> str:
    if not suffix:
        raise ValueError("Sidecar suffix must be non-empty")
    if not video_path.endswith(".mp4"):
        raise ValueError(f"Expected .mp4 video path, got {video_path}")
    return video_path[: -len(".mp4")] + suffix


def _to_preprocessed_video_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous float32 video tensor normalized to [-1, 1]."""
    tensor = tensor.contiguous()
    if torch.is_floating_point(tensor):
        tensor = tensor.float()
        if tensor.numel() == 0:
            return tensor
        min_value = float(tensor.amin())
        max_value = float(tensor.amax())
        if min_value >= -0.0001 and max_value <= 1.0001:
            return tensor.mul(2.0).sub(1.0)
        if min_value >= -1.0001 and max_value <= 1.0001:
            return tensor
        return tensor.div(127.5).sub(1.0)
    return tensor.float().div(127.5).sub(1.0)


class SurgicalTransferJSONDataset(SurgicalVideoJSONDataset):
    """Iterable surgical transfer dataset with RGB targets and control videos.

    Expected local layout for each manifest entry ``foo.mp4``::

        foo.mp4
        foo.json
        foo.depth.mp4
        foo.seg.mp4        # configurable via seg_suffix
        foo.blur.mp4       # optional, configurable via blur_suffix

    ``edge`` controls are computed from the RGB target clip on the fly. ``blur``
    first tries to load a precomputed sidecar and falls back to on-the-fly blur
    when the sidecar is absent or unreadable. ``depth`` and ``seg`` controls
    are loaded from sidecar MP4s using the exact same frame range as the RGB
    target. Samples are returned as preprocessed
    ``video=[control, target]`` float tensors in ``[-1, 1]`` with shared temporal
    mRoPE positions.
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
        temporal_compression_factor: int = 4,
        frame_selection_mode: str = "random",
        num_decode_threads: int = 2,
        control_modalities: dict[str, float] | None = None,
        depth_suffix: str = ".depth.mp4",
        blur_suffix: str = ".blur.mp4",
        seg_suffix: str = ".seg.mp4",
        edge_use_random: bool = True,
        blur_use_random: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
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
            conditioning_config=None,
            temporal_compression_factor=temporal_compression_factor,
            frame_selection_mode=frame_selection_mode,
            num_decode_threads=num_decode_threads,
        )
        if kwargs:
            log.info(f"Unknown kwargs for SurgicalTransferJSONDataset: {kwargs}")
        self.control_modalities = _normalize_weights(control_modalities)
        self.depth_suffix = depth_suffix
        self.blur_suffix = blur_suffix
        self.seg_suffix = seg_suffix
        self._edge_augmentor = AddControlInputEdge(
            input_keys=["video"], output_keys=["control_input_edge"], use_random=edge_use_random
        )
        self._blur_augmentor = AddControlInputBlur(
            input_keys=["video"], output_keys=["control_input_blur"], use_random=blur_use_random
        )
        self._depth_augmentor = AddControlInputDepth(input_keys=["video"], output_keys=["control_input_depth"])
        self._seg_augmentor = AddControlInputSeg(input_keys=["video"], output_keys=["control_input_seg"])
        log.info(f"SurgicalTransferJSONDataset control modality weights: {self.control_modalities}")

    def _select_modality(self, rng: random.Random) -> str:
        modalities = list(self.control_modalities)
        weights = [self.control_modalities[key] for key in modalities]
        return rng.choices(modalities, weights=weights, k=1)[0]

    def _decode_sidecar_video(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        scale_flags: str,
    ) -> torch.Tensor:
        target_h, target_w = self.video_size
        frame_size = target_h * target_w * 3
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-threads",
            str(self.num_decode_threads),
            "-filter_threads",
            str(self.num_decode_threads),
            "-filter_complex_threads",
            str(self.num_decode_threads),
            "-i",
            video_path,
            "-pix_fmt",
            "rgb24",
            "-vf",
            f"scale={target_w}:{target_h}:flags={scale_flags}",
            "-f",
            "rawvideo",
            "-vsync",
            "0",
            "-",
        ]
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=-1,
        )
        frames: list[np.ndarray] = []
        try:
            assert process.stdout is not None
            idx = 0
            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    if len(raw_frame) != 0:
                        raise ValueError(f"Incomplete frame from {video_path}: {len(raw_frame)} bytes")
                    break
                if idx >= start_frame and idx <= end_frame:
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((target_h, target_w, 3))
                    frames.append(frame)
                if idx > end_frame:
                    break
                idx += 1
        finally:
            if process.stdout is not None:
                process.stdout.close()
            process.wait()

        if len(frames) != self.num_frames:
            raise ValueError(
                f"Decoded {len(frames)} frames from {video_path}, expected {self.num_frames} "
                f"(start={start_frame}, end={end_frame})"
            )
        video_np = np.stack(frames, axis=0)
        target_t = (video_np.shape[0] - 1) // self.temporal_compression_factor * self.temporal_compression_factor + 1
        video_np = video_np[:target_t]
        video_np = np.transpose(video_np, (3, 0, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(video_np)).to(torch.uint8)

    def _build_control(self, video_path: str, video: torch.Tensor, start_frame: int, end_frame: int, modality: str):
        if modality == "edge":
            data = self._edge_augmentor({"video": video})
            return data["control_input_edge"]
        if modality == "blur":
            if self.blur_suffix:
                sidecar_path = _resolve_sidecar_path(video_path, self.blur_suffix)
                if Path(sidecar_path).exists():
                    try:
                        return self._decode_sidecar_video(sidecar_path, start_frame, end_frame, scale_flags="bicubic")
                    except Exception as e:
                        log.warning(
                            f"Failed to load precomputed blur sidecar {sidecar_path}; "
                            f"falling back to on-the-fly blur: {e}",
                            rank0_only=False,
                        )
            data = self._blur_augmentor({"video": video})
            return data["control_input_blur"]
        if modality == "depth":
            sidecar_path = _resolve_sidecar_path(video_path, self.depth_suffix)
            if not Path(sidecar_path).exists():
                raise FileNotFoundError(f"Depth sidecar not found: {sidecar_path}")
            depth = self._decode_sidecar_video(sidecar_path, start_frame, end_frame, scale_flags="bicubic")
            data = self._depth_augmentor({"video": video, "depth": depth})
            return data["control_input_depth"]
        if modality == "seg":
            sidecar_path = _resolve_sidecar_path(video_path, self.seg_suffix)
            if not Path(sidecar_path).exists():
                raise FileNotFoundError(f"{modality} sidecar not found: {sidecar_path}")
            segmentation = self._decode_sidecar_video(sidecar_path, start_frame, end_frame, scale_flags="neighbor")
            data = self._seg_augmentor({"video": video, "segmentation": segmentation})
            return data["control_input_seg"]
        raise ValueError(f"Unsupported transfer modality: {modality}")

    def process_one_sample(self, video_path: str, rng: random.Random) -> dict | None:
        sample_t0 = time.monotonic()
        step_times: dict[str, float] = {}
        try:
            t0 = time.monotonic()
            video_info = get_video_metadata(video_path)
            original_fps = video_info["fps"]
            total_frames = video_info["total_frames"]
            start_frame, end_frame = self._select_frame_range(total_frames, rng)
            step_times["metadata"] = time.monotonic() - t0

            t0 = time.monotonic()
            video = self._decode_video(video_path, start_frame, end_frame)
            step_times["decode_rgb"] = time.monotonic() - t0

            modality = self._select_modality(rng)
            t0 = time.monotonic()
            control = self._build_control(video_path, video, start_frame, end_frame, modality)
            step_times[f"control_{modality}"] = time.monotonic() - t0

            if control.shape != video.shape:
                raise ValueError(
                    f"Control shape {tuple(control.shape)} does not match video shape {tuple(video.shape)}"
                )
            t0 = time.monotonic()
            control = _to_preprocessed_video_tensor(control)
            video = _to_preprocessed_video_tensor(video)
            step_times["normalize"] = time.monotonic() - t0

            t0 = time.monotonic()
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
            step_times["caption_tokenize"] = time.monotonic() - t0

            t0 = time.monotonic()
            image_size = torch.tensor([target_h, target_w, target_h, target_w], dtype=torch.float32)
            sample = dict(
                __key__=Path(video_path).stem,
                __url__=video_path,
                fps=original_fps,
                n_orig_video_frames=total_frames,
                frame_start=start_frame,
                frame_end=end_frame,
                num_frames=video.shape[1],
                video=[control, video],
                num_multiplier=1,
                conditioning_fps=cond_fps,
                padding_mask=torch.zeros((1, target_h, target_w), dtype=torch.float32),
                image_size=[image_size, image_size.clone()],
                ai_caption=caption,
                sampled_caption_style=caption_key,
                text_token_ids=torch.tensor(text_ids),
                transfer_modality=modality,
                sequence_plan=SequencePlan(
                    has_text=True,
                    has_vision=True,
                    condition_frame_indexes_vision=[],
                    share_vision_temporal_positions=True,
                ),
            )
            step_times["format_output"] = time.monotonic() - t0
            sample["_sample_time"] = time.monotonic() - sample_t0
            sample["_aug_time"] = sum(step_times.values())
            sample["_aug_step_times"] = step_times
            return sample
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load transfer video {video_path} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            return None


def get_surgical_transfer_json_dataset(
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
    temporal_compression_factor: int = 4,
    frame_selection_mode: str = "random",
    num_decode_threads: int = 2,
    control_modalities: dict[str, float] | None = None,
    depth_suffix: str = ".depth.mp4",
    blur_suffix: str = ".blur.mp4",
    seg_suffix: str = ".seg.mp4",
    **kwargs: Any,
) -> SurgicalTransferJSONDataset:
    return SurgicalTransferJSONDataset(
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
        temporal_compression_factor=temporal_compression_factor,
        frame_selection_mode=frame_selection_mode,
        num_decode_threads=num_decode_threads,
        control_modalities=control_modalities,
        depth_suffix=depth_suffix,
        blur_suffix=blur_suffix,
        seg_suffix=seg_suffix,
        **kwargs,
    )
