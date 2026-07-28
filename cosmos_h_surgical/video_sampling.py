# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import io
import math
from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

_SUPPORTED_VIDEO_EXTENSIONS = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})
_JPEG_QUALITY = 85
_MAX_FRAME_EDGE = 768


@dataclass(frozen=True, slots=True)
class SampledVideo:
    data_urls: tuple[str, ...]
    timestamps_seconds: tuple[float, ...]
    duration_seconds: float
    source_fps: float

    @property
    def effective_sample_fps(self) -> float:
        return len(self.data_urls) / self.duration_seconds


def calculate_sample_timestamps(
    duration_seconds: float,
    *,
    frame_count: int = 10,
    min_fps: float = 1.0,
    max_frames: int = 32,
) -> tuple[float, ...]:
    """Return temporal-bin centers spanning the complete video duration."""
    if duration_seconds <= 0:
        raise ValueError("Video duration must be positive")
    if frame_count <= 0:
        raise ValueError("Video frame count must be positive")
    if min_fps <= 0:
        raise ValueError("Video minimum sampling FPS must be positive")
    if max_frames <= 0:
        raise ValueError("Video maximum frame count must be positive")

    sample_count = max(frame_count, math.ceil(duration_seconds * min_fps))
    if sample_count > max_frames:
        raise ValueError(
            f"Video requires {sample_count} frames to satisfy frame_count={frame_count} "
            f"and min_fps={min_fps}, exceeding max_frames={max_frames}. "
            "Increase --video-max-frames or use a shorter video."
        )

    bin_width = duration_seconds / sample_count
    return tuple((index + 0.5) * bin_width for index in range(sample_count))


def _frame_to_jpeg_data_url(frame: Any) -> str:
    image = Image.fromarray(frame.detach().cpu().numpy())
    if max(image.size) > _MAX_FRAME_EDGE:
        image.thumbnail((_MAX_FRAME_EDGE, _MAX_FRAME_EDGE), Image.Resampling.LANCZOS)

    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    payload = b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _default_decoder_factory(source: Path, **kwargs: Any) -> Any:
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(source, **kwargs)


def sample_video_frames(
    path: str | Path,
    *,
    frame_count: int = 10,
    min_fps: float = 1.0,
    max_frames: int = 32,
    decoder_factory: Callable[..., Any] = _default_decoder_factory,
) -> SampledVideo:
    """Decode uniformly distributed chronological frames from a local video."""
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if video_path.suffix.lower() not in _SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension {video_path.suffix!r}; expected one of: {supported}")

    decoder = decoder_factory(
        video_path,
        dimension_order="NHWC",
        device="cpu",
        seek_mode="exact",
    )
    metadata = decoder.metadata
    begin_seconds = float(metadata.begin_stream_seconds or 0.0)
    end_seconds = metadata.end_stream_seconds
    duration_seconds = metadata.duration_seconds
    if end_seconds is not None:
        duration_seconds = float(end_seconds) - begin_seconds
    if duration_seconds is None or duration_seconds <= 0:
        raise ValueError(f"Unable to determine a positive duration for {video_path}")

    source_fps = metadata.average_fps
    if source_fps is None or source_fps <= 0:
        raise ValueError(f"Unable to determine a positive source FPS for {video_path}")
    source_fps = float(source_fps)
    if source_fps < min_fps:
        raise ValueError(f"Source video FPS {source_fps:.3f} is below the requested minimum sampling FPS {min_fps:.3f}")

    relative_timestamps = calculate_sample_timestamps(
        duration_seconds,
        frame_count=frame_count,
        min_fps=min_fps,
        max_frames=max_frames,
    )
    if len(relative_timestamps) > len(decoder):
        raise ValueError(
            f"Video contains {len(decoder)} frames, fewer than the {len(relative_timestamps)} requested samples"
        )

    absolute_timestamps = [begin_seconds + timestamp for timestamp in relative_timestamps]
    frames = decoder.get_frames_played_at(absolute_timestamps)
    actual_timestamps = tuple(float(value) - begin_seconds for value in frames.pts_seconds.tolist())
    if len(set(actual_timestamps)) != len(actual_timestamps):
        raise ValueError(f"Video sampling produced duplicate source frames for {video_path}")

    return SampledVideo(
        data_urls=tuple(_frame_to_jpeg_data_url(frame) for frame in frames.data),
        timestamps_seconds=actual_timestamps,
        duration_seconds=duration_seconds,
        source_fps=source_fps,
    )
