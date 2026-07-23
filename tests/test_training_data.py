# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest

from cosmos_h_surgical.data.manifests import (
    VideoMetadata,
    discover_training_videos,
    load_manifest_entries,
    prepare_training_manifest,
    validate_training_manifests,
)
from cosmos_h_surgical.data.surgical_transfer_json_dataset import _normalize_weights, _resolve_sidecar_path
from cosmos_h_surgical.data.surgical_video_json_dataset import _as_list, _parse_factors


def _write_caption(video: Path) -> None:
    video.with_suffix(".json").write_text(json.dumps({"caption_json": {"description": "Surgical action."}}))


def test_development_manifest_helpers_preserve_parallel_source_lists() -> None:
    assert _as_list("/data/a, /data/b") == ["/data/a", "/data/b"]
    assert _parse_factors("1.0, 0.5") == [1.0, 0.5]


def test_development_manifest_helpers_tolerate_shell_quote_characters() -> None:
    assert _as_list("\u201d/data/a, /data/b\u201d") == ["/data/a", "/data/b"]
    assert _parse_factors("\u201d1.0,1.0\u201d") == [1.0, 1.0]
    assert _parse_factors('"1.0,0.5"') == [1.0, 0.5]


def test_parse_factors_reports_the_invalid_environment_value() -> None:
    with pytest.raises(ValueError, match="expected comma-separated numbers"):
        _parse_factors("not-a-number")


def test_prepare_manifest_excludes_transfer_sidecars(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    videos = root / "videos"
    videos.mkdir(parents=True)
    target = videos / "clip_000001.mp4"
    target.touch()
    _write_caption(target)
    for suffix in (".blur.mp4", ".depth.mp4", ".seg.mp4"):
        videos.joinpath(f"clip_000001{suffix}").touch()

    output = root / "manifests" / "train.json"
    count = prepare_training_manifest(root, output, ["videos/**/*.mp4"])

    assert count == 1
    assert load_manifest_entries(output) == ["videos/clip_000001.mp4"]
    assert discover_training_videos(root, ["videos/**/*.mp4"]) == [target.resolve()]


def test_prepare_manifest_requires_captions(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "clip.mp4").touch()

    with pytest.raises(ValueError, match="Missing json caption"):
        prepare_training_manifest(root, root / "train.json", ["*.mp4"])


def test_prepare_manifest_keeps_symlink_paths_portable(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    root = tmp_path / "dataset"
    root.mkdir()
    video = root / "clip.mp4"
    video.symlink_to(source)
    _write_caption(video)

    output = root / "train.json"
    prepare_training_manifest(root, output, ["*.mp4"])

    assert load_manifest_entries(output) == ["clip.mp4"]
    report = validate_training_manifests(
        root,
        [output],
        mode="predict",
        media_probe=lambda _: VideoMetadata(width=832, height=480, frames=93, fps=16.0),
    )
    assert report.ok


def test_validate_predict_manifest_with_portable_paths(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    video = root / "clip.mp4"
    video.touch()
    _write_caption(video)
    manifest = root / "train.json"
    manifest.write_text(json.dumps({"training": [{"video": "clip.mp4"}]}))

    report = validate_training_manifests(
        root,
        [manifest],
        mode="predict",
        media_probe=lambda _: VideoMetadata(width=832, height=480, frames=93, fps=16.0),
    )

    assert report.ok
    assert report.videos == 1


def test_validate_transfer_requires_aligned_depth_and_seg(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    video = root / "clip.mp4"
    video.touch()
    _write_caption(video)
    (root / "clip.depth.mp4").touch()
    manifest = root / "train.json"
    manifest.write_text(json.dumps({"training": ["clip.mp4"]}))

    def metadata(path: Path) -> VideoMetadata:
        frames = 92 if path.name.endswith(".depth.mp4") else 93
        return VideoMetadata(width=832, height=480, frames=frames, fps=16.0)

    report = validate_training_manifests(root, [manifest], mode="transfer", media_probe=metadata)

    assert not report.ok
    assert any("misaligned depth control" in error for error in report.errors)
    assert any("seg control not found" in error for error in report.errors)
    assert not any("blur control not found" in error for error in report.errors)


def test_transfer_loader_supports_only_public_controls() -> None:
    assert _normalize_weights({"edge": 2, "blur": 2, "depth": 2, "seg": 2}) == {
        "edge": 0.25,
        "blur": 0.25,
        "depth": 0.25,
        "seg": 0.25,
    }
    with pytest.raises(ValueError, match="Unsupported transfer modalities"):
        _normalize_weights({"flow": 1})
    assert _resolve_sidecar_path("clip.mp4", ".depth.mp4") == "clip.depth.mp4"
