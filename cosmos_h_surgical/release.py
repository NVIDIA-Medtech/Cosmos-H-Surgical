# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cosmos_h_surgical.__about__ import __version__
from cosmos_h_surgical.provenance import FRAMEWORK_REPOSITORY, FRAMEWORK_REVISION

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_INTERNAL_MARKERS = (
    "/home/",
    "/lustre/",
    "/healthcareeng_",
    "gitlab-master.nvidia.com",
    "s3://bucket1/",
)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("release manifest must contain a JSON object")
    return value


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def validate_manifest(path: Path) -> list[str]:
    manifest = load_manifest(path)
    errors: list[str] = []

    if manifest.get("schema_version") != 2:
        errors.append("schema_version must be 2")

    release = manifest.get("release")
    if not isinstance(release, dict):
        errors.append("release must be an object")
    elif release.get("version") != f"v{__version__}":
        errors.append(f"release.version must be v{__version__}")

    framework = manifest.get("framework")
    if not isinstance(framework, dict):
        errors.append("framework must be an object")
    else:
        revision = framework.get("revision")
        if revision != FRAMEWORK_REVISION or not isinstance(revision, str) or not _SHA40.fullmatch(revision):
            errors.append(f"framework.revision must be the pinned 40-character SHA {FRAMEWORK_REVISION}")
        if framework.get("repository") != FRAMEWORK_REPOSITORY:
            errors.append(f"framework.repository must be {FRAMEWORK_REPOSITORY}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")

    for text in _iter_strings(manifest):
        marker = next((item for item in _INTERNAL_MARKERS if item in text), None)
        if marker is not None:
            errors.append(f"manifest contains forbidden internal marker {marker!r}")

    return errors
