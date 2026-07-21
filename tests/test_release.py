# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import tomllib
from pathlib import Path

from cosmos_h_surgical.checkpoints import load_checkpoint_registry
from cosmos_h_surgical.release import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release-manifest.json"
PUBLIC_TEXT_FILES = [
    ROOT / "README.md",
    ROOT / "UPSTREAM.md",
    ROOT / "NOTICE",
    ROOT / "ATTRIBUTIONS.md",
    ROOT / "pyproject.toml",
    ROOT / "release-manifest.json",
    *sorted(path for path in (ROOT / "cosmos_h_surgical").rglob("*.py") if path.name != "release.py"),
    *sorted((ROOT / "examples").rglob("*.md")),
    *sorted((ROOT / "examples").rglob("*.toml")),
]
FORBIDDEN_INTERNAL_MARKERS = (
    "/home/",
    "/lustre/",
    "/healthcareeng_",
    "gitlab-master.nvidia.com",
    "s3://bucket1/",
)


def test_development_manifest_is_valid_and_has_no_artifacts() -> None:
    assert validate_manifest(MANIFEST) == []
    assert load_checkpoint_registry(MANIFEST) == {}


def test_manifest_has_no_internal_or_moving_framework_references() -> None:
    text = MANIFEST.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "/lustre/" not in text
    assert "gitlab-master.nvidia.com" not in text
    assert '"revision": "main"' not in text


def test_public_sources_have_no_internal_references() -> None:
    for path in PUBLIC_TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_INTERNAL_MARKERS:
            assert marker not in text, f"{path.relative_to(ROOT)} contains {marker!r}"


def test_manifest_is_formatted_json() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = json.dumps(value, indent=2) + "\n"
    assert MANIFEST.read_text(encoding="utf-8") == expected


def test_dependency_contract_has_one_full_environment_per_cuda_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "cosmos-framework[train]" in project["project"]["dependencies"]
    assert "optional-dependencies" not in project["project"]

    groups = project["dependency-groups"]
    assert "torch==2.10.0+cu128" in groups["cu128"]
    assert "transformer-engine==2.12.0+cu128.torch210" in groups["cu128"]
    assert "torch==2.10.0+cu130" in groups["cu130"]
    assert "transformer-engine==2.12.0+cu130.torch210" in groups["cu130"]

    assert project["tool"]["uv"]["conflicts"] == [[{"group": "cu128"}, {"group": "cu130"}]]
    assert project["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch"}
