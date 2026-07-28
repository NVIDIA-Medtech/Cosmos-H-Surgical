# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "cosmos-h-surgical-assets"
EXPECTED_STEMS = (
    "aspiration",
    "clipping",
    "coagulation",
    "dissection",
    "knotting",
    "needleGrasping",
    "needlePuncture",
    "packing",
    "suturePulling",
    "tissueRetraction",
)


def test_toy_dataset_manifest_and_sidecars() -> None:
    manifest = json.loads((DATASET_ROOT / "manifests" / "train.json").read_text())
    assert manifest == {"training": [{"video": f"videos/{stem}.mp4"} for stem in EXPECTED_STEMS]}

    for stem in EXPECTED_STEMS:
        for suffix in (".mp4", ".blur.mp4", ".depth.mp4", ".seg.mp4"):
            path = DATASET_ROOT / "videos" / f"{stem}{suffix}"
            assert path.is_file()
            assert path.stat().st_size > 0


def test_toy_dataset_captions_expose_only_caption_json() -> None:
    for stem in EXPECTED_STEMS:
        caption_path = DATASET_ROOT / "videos" / f"{stem}.json"
        caption = json.loads(caption_path.read_text())
        assert set(caption) == {"caption_json"}
        assert isinstance(caption["caption_json"], dict)
        assert caption["caption_json"]

        serialized = caption_path.read_text()
        assert "/home/" not in serialized
        assert "/workspace/" not in serialized
