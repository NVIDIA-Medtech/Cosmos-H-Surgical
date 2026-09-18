# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import tomllib
from pathlib import Path

from cosmos_h_surgical.checkpoints import (
    DEFAULT_MODEL_KEY,
    DMD2_HF_SUBDIRECTORY,
    DMD2_MODEL_CONFIG_PATH,
    DMD2_MODEL_KEY,
    MODEL_CONFIG_PATH,
    load_checkpoint_registry,
)
from cosmos_h_surgical.release import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release-manifest.json"
PUBLIC_TEXT_FILES = [
    ROOT / "README.md",
    ROOT / "UPSTREAM.md",
    ROOT / "LICENSE.weights",
    ROOT / "NOTICE",
    ROOT / "ATTRIBUTIONS.md",
    ROOT / "pyproject.toml",
    ROOT / "release-manifest.json",
    *sorted(path for path in (ROOT / "cosmos_h_surgical").rglob("*.py") if path.name != "release.py"),
    *sorted((ROOT / "docs").rglob("*.md")),
    *sorted((ROOT / "examples").rglob("*.md")),
    *sorted((ROOT / "examples").rglob("*.toml")),
    *sorted((ROOT / "scripts").rglob("*.sh")),
]
FORBIDDEN_INTERNAL_MARKERS = (
    "/home/",
    "/lustre/",
    "/healthcareeng_",
    "gitlab-master.nvidia.com",
    "s3://bucket1/",
)


def test_release_manifest_is_final_and_registers_public_model() -> None:
    assert validate_manifest(MANIFEST) == []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["release"]["status"] == "released"
    assert manifest["framework"]["status"] == "pinned-release"
    assert manifest["license"]["id"] == "OpenMDW-1.1"
    assert manifest["weights_license"]["id"] == "OpenMDW-1.1"
    registry = load_checkpoint_registry(MANIFEST)
    assert set(registry) == {DEFAULT_MODEL_KEY, DMD2_MODEL_KEY}
    assert registry[DEFAULT_MODEL_KEY].path == "."
    assert registry[DEFAULT_MODEL_KEY].revision == "v0.3.1"
    assert registry[DEFAULT_MODEL_KEY].license_id == "OpenMDW-1.1"
    assert registry[DMD2_MODEL_KEY].path == DMD2_HF_SUBDIRECTORY
    assert registry[DMD2_MODEL_KEY].revision == "v0.3.1"
    assert registry[DMD2_MODEL_KEY].license_id == "OpenMDW-1.1"


def test_packaged_model_config_is_portable() -> None:
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = config["model"]["config"]
    tokenizer = model_config["tokenizer"]

    assert model_config["enable_input_bias"] is True
    assert tokenizer["bucket_name"] == "bucket"
    assert tokenizer["vae_path"] == "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
    text = MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    for marker in FORBIDDEN_INTERNAL_MARKERS:
        assert marker not in text


def test_packaged_dmd2_model_config_is_portable_and_four_step() -> None:
    config = json.loads(DMD2_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = config["model"]["config"]

    assert model_config["fixed_step_sampler_config"] == {
        "sample_type": "sde",
        "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
    }
    for marker in FORBIDDEN_INTERNAL_MARKERS:
        assert marker not in DMD2_MODEL_CONFIG_PATH.read_text(encoding="utf-8")


def test_public_dmd2_tutorial_does_not_document_internal_control_variant() -> None:
    tutorial = (ROOT / "docs" / "dmd2_distillation.md").read_text(encoding="utf-8")
    assert "seg_tool" not in tutorial.lower()


def test_public_dmd2_recipe_does_not_override_internal_control_suffix() -> None:
    recipe = (ROOT / "cosmos_h_surgical" / "configs" / "transfer_dmd2.py").read_text(encoding="utf-8")
    assert "seg_tool_suffix" not in recipe


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
