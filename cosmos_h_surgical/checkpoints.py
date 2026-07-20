# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosmos_h_surgical.release import load_manifest


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
            raise KeyError("No Cosmos 3 checkpoint has been published for v0.3.0") from error
        raise KeyError(f"Unknown model key {model_key!r}; available: {sorted(registry)}") from error
