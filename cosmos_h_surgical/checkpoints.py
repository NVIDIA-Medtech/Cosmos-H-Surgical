# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosmos_h_surgical.release import load_manifest

DEFAULT_MODEL_KEY = "Cosmos-H-Surgical"
DMD2_MODEL_KEY = "Cosmos-H-Surgical-Transfer-DMD2-4Step"
DEFAULT_HF_REPOSITORY = "nvidia/Cosmos-H-Surgical"
DEFAULT_HF_REVISION = "v0.3.1"
DMD2_HF_SUBDIRECTORY = "dmd2-transfer-480p-4step"
HF_REPOSITORY_ENV = "COSMOS_H_SURGICAL_HF_REPOSITORY"
HF_REVISION_ENV = "COSMOS_H_SURGICAL_HF_REVISION"
BASE_MODEL_CONFIG_PATH = Path(__file__).parent / "configs" / "cosmos_h_surgical_v0.3.0.json"
DMD2_MODEL_CONFIG_PATH = Path(__file__).parent / "configs" / "cosmos_h_surgical_transfer_dmd2_4step_v0.3.1.json"
# Retain the original constant for downstream callers that import it.
MODEL_CONFIG_PATH = BASE_MODEL_CONFIG_PATH


@dataclass(frozen=True)
class CheckpointArtifact:
    model_key: str
    path: str
    revision: str
    sha256: str
    license_id: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CheckpointArtifact:
        return cls(
            model_key=str(value["model_key"]),
            path=str(value["path"]),
            revision=str(value["revision"]),
            sha256=str(value["sha256"]),
            license_id=str(value["license_id"]),
        )


def load_checkpoint_registry(manifest_path: Path) -> dict[str, CheckpointArtifact]:
    manifest = load_manifest(manifest_path)
    return {
        artifact.model_key: artifact
        for artifact in (CheckpointArtifact.from_dict(item) for item in manifest["artifacts"])
    }


def resolve_checkpoint(model_key: str, manifest_path: Path) -> CheckpointArtifact:
    registry = load_checkpoint_registry(manifest_path)
    try:
        return registry[model_key]
    except KeyError as error:
        if not registry:
            raise KeyError("No Cosmos 3 checkpoint has been published for v0.3.1") from error
        raise KeyError(f"Unknown model key {model_key!r}; available: {sorted(registry)}") from error


def register_checkpoint_alias() -> None:
    """Register the public model name with the pinned framework inference CLI."""
    from cosmos_framework.inference.args import _CHECKPOINTS, MODEL_MEMORY_BYTES_BY_SIZE
    from cosmos_framework.inference.common.args import CheckpointConfig
    from cosmos_framework.utils.checkpoint_db import CheckpointDirHf

    repository = os.environ.get(HF_REPOSITORY_ENV, DEFAULT_HF_REPOSITORY)
    revision = os.environ.get(HF_REVISION_ENV, DEFAULT_HF_REVISION)
    aliases = (
        (DEFAULT_MODEL_KEY, BASE_MODEL_CONFIG_PATH, ""),
        (DMD2_MODEL_KEY, DMD2_MODEL_CONFIG_PATH, DMD2_HF_SUBDIRECTORY),
    )
    for model_key, config_path, subdirectory in aliases:
        _CHECKPOINTS[model_key] = CheckpointConfig(
            model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
            config_file=str(config_path),
            # The inference path always uses checkpoint_hf. This URI satisfies the
            # upstream registry schema without introducing a private object-store path.
            s3_uri=f"hf://{repository}/{subdirectory}" if subdirectory else f"hf://{repository}",
            hf=CheckpointDirHf(
                repository=repository,
                revision=revision,
                subdirectory=subdirectory,
            ),
        )
