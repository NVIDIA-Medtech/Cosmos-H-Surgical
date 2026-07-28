# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import av

CONTROL_SUFFIXES = (".blur.mp4", ".depth.mp4", ".seg.mp4")
SUPPORTED_CONTROL_MODALITIES = frozenset({"edge", "blur", "depth", "seg"})


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    frames: int
    fps: float


@dataclass(frozen=True)
class ValidationReport:
    manifests: int
    videos: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_control_sidecar(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in CONTROL_SUFFIXES)


def discover_training_videos(
    dataset_dir: Path,
    video_patterns: Sequence[str],
    *,
    caption_format: str = "json",
    require_captions: bool = True,
) -> list[Path]:
    """Resolve target-video globs while excluding transfer-control sidecars."""
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    if caption_format not in {"json", "text"}:
        raise ValueError(f"Unsupported caption format: {caption_format}")
    if not video_patterns:
        raise ValueError("At least one video pattern is required")

    videos = {
        Path(os.path.abspath(match))
        for pattern in video_patterns
        for match in root.glob(pattern)
        if match.is_file() and not _is_control_sidecar(match)
    }
    outside_root = [path for path in videos if not path.is_relative_to(root)]
    if outside_root:
        raise ValueError(f"Video pattern escaped the dataset directory: {outside_root[0]}")
    if not videos:
        raise ValueError(f"No target videos matched: {', '.join(video_patterns)}")

    suffix = ".json" if caption_format == "json" else ".txt"
    missing_captions = [path for path in videos if not path.with_suffix(suffix).is_file()]
    if require_captions and missing_captions:
        preview = ", ".join(str(path.relative_to(root)) for path in sorted(missing_captions)[:5])
        raise ValueError(f"Missing {caption_format} caption sidecars for: {preview}")
    return sorted(videos)


def prepare_training_manifest(
    dataset_dir: Path,
    output_path: Path,
    video_patterns: Sequence[str],
    *,
    caption_format: str = "json",
    require_captions: bool = True,
    force: bool = False,
) -> int:
    """Write a development-format manifest with portable relative video paths."""
    root = dataset_dir.expanduser().resolve()
    videos = discover_training_videos(
        root,
        video_patterns,
        caption_format=caption_format,
        require_captions=require_captions,
    )
    output = output_path.expanduser().resolve()
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists; pass --force to replace it: {output_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"training": [{"video": path.relative_to(root).as_posix()} for path in videos]}

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, output)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return len(videos)


def load_manifest_entries(path: Path) -> list[str]:
    """Load target video paths from the accepted development manifest variants."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Failed to read manifest {path}: {error}") from error
    if isinstance(payload, dict):
        if "training" not in payload:
            raise ValueError(f"Manifest {path} must contain a 'training' list")
        entries = payload["training"]
    else:
        entries = payload
    if not isinstance(entries, list):
        raise ValueError(f"Manifest {path} must contain a list of entries")

    videos: list[str] = []
    for index, entry in enumerate(entries):
        video = entry.get("video") if isinstance(entry, dict) else entry
        if not isinstance(video, str) or not video.strip():
            raise ValueError(f"Manifest {path} entry {index} must provide a non-empty video path")
        videos.append(video)
    return videos


def probe_video(path: Path) -> VideoMetadata:
    """Read video stream metadata with the PyAV dependency."""
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frame_rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            if frame_rate is None:
                raise ValueError("video stream does not report a frame rate")
            frames = int(stream.frames or 0)
            if frames <= 0:
                frames = sum(1 for _ in container.decode(stream))
            return VideoMetadata(
                width=int(stream.width),
                height=int(stream.height),
                frames=frames,
                fps=float(frame_rate),
            )
    except (IndexError, OSError, TypeError, ValueError, ZeroDivisionError, av.error.FFmpegError) as error:
        raise ValueError(f"Failed to probe video {path}: {error}") from error


def _resolve_entry(dataset_dir: Path, value: str, *, require_relative: bool) -> Path:
    entry = Path(value).expanduser()
    if require_relative and entry.is_absolute():
        raise ValueError("absolute paths are not portable")
    normalized = Path(os.path.abspath(entry if entry.is_absolute() else dataset_dir / entry))
    if require_relative and not normalized.is_relative_to(dataset_dir):
        raise ValueError("relative path escapes the dataset directory")
    return normalized


def _compare_control(target: VideoMetadata, control: VideoMetadata) -> str | None:
    mismatches: list[str] = []
    if (target.width, target.height) != (control.width, control.height):
        mismatches.append(f"size {control.width}x{control.height} != {target.width}x{target.height}")
    if target.frames != control.frames:
        mismatches.append(f"frames {control.frames} != {target.frames}")
    if abs(target.fps - control.fps) > 1.0e-3:
        mismatches.append(f"fps {control.fps:g} != {target.fps:g}")
    return ", ".join(mismatches) or None


def validate_training_manifests(
    dataset_dir: Path,
    manifest_paths: Iterable[Path],
    *,
    mode: str,
    caption_format: str = "json",
    control_modalities: Iterable[str] = ("edge", "blur", "depth", "seg"),
    minimum_frames: int = 93,
    require_relative: bool = True,
    media_probe: Callable[[Path], VideoMetadata] | None = probe_video,
) -> ValidationReport:
    """Validate manifests, captions, and aligned transfer-control sidecars."""
    if mode not in {"predict", "transfer"}:
        raise ValueError(f"Unsupported training mode: {mode}")
    if caption_format not in {"json", "text"}:
        raise ValueError(f"Unsupported caption format: {caption_format}")
    modalities = tuple(dict.fromkeys(control_modalities))
    unsupported = set(modalities) - SUPPORTED_CONTROL_MODALITIES
    if unsupported:
        raise ValueError(f"Unsupported control modalities: {sorted(unsupported)}")

    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    errors: list[str] = []
    seen: set[Path] = set()
    video_count = 0
    manifests = list(manifest_paths)
    for manifest in manifests:
        manifest_path = manifest.expanduser().resolve()
        try:
            entries = load_manifest_entries(manifest_path)
        except ValueError as error:
            errors.append(str(error))
            continue
        for index, entry in enumerate(entries):
            label = f"{manifest_path.name}[{index}]"
            try:
                video_path = _resolve_entry(root, entry, require_relative=require_relative)
            except ValueError as error:
                errors.append(f"{label}: {error}: {entry}")
                continue
            if video_path in seen:
                errors.append(f"{label}: duplicate video entry: {entry}")
            seen.add(video_path)
            video_count += 1
            if _is_control_sidecar(video_path):
                errors.append(f"{label}: control sidecar listed as a target video: {entry}")
            if not video_path.is_file():
                errors.append(f"{label}: target video not found: {entry}")
                continue

            caption_path = video_path.with_suffix(".json" if caption_format == "json" else ".txt")
            if not caption_path.is_file():
                errors.append(f"{label}: caption sidecar not found: {caption_path.name}")
            elif caption_format == "json":
                try:
                    caption = json.loads(caption_path.read_text())
                    if not isinstance(caption, dict) or not caption.get("caption_json"):
                        errors.append(f"{label}: caption JSON must contain a non-empty 'caption_json' field")
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(f"{label}: invalid caption JSON {caption_path.name}: {error}")

            target_metadata: VideoMetadata | None = None
            if media_probe is not None:
                try:
                    target_metadata = media_probe(video_path)
                    if target_metadata.frames < minimum_frames:
                        errors.append(
                            f"{label}: target has {target_metadata.frames} frames; at least {minimum_frames} required"
                        )
                except ValueError as error:
                    errors.append(f"{label}: {error}")

            if mode != "transfer":
                continue
            for modality in modalities:
                if modality == "edge":
                    continue
                suffix = f".{modality}.mp4"
                control_path = video_path.with_name(f"{video_path.stem}{suffix}")
                if modality == "blur" and not control_path.is_file():
                    continue
                if not control_path.is_file():
                    errors.append(f"{label}: {modality} control not found: {control_path.name}")
                    continue
                if media_probe is not None and target_metadata is not None:
                    try:
                        control_metadata = media_probe(control_path)
                        mismatch = _compare_control(target_metadata, control_metadata)
                        if mismatch:
                            errors.append(f"{label}: misaligned {modality} control: {mismatch}")
                    except ValueError as error:
                        errors.append(f"{label}: {error}")

    return ValidationReport(manifests=len(manifests), videos=video_count, errors=tuple(errors))
