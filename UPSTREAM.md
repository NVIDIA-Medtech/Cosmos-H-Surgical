# Cosmos Framework Provenance

Cosmos-H-Surgical uses the public NVIDIA Cosmos Framework as a read-only
dependency. This repository does not patch installed framework files or depend
on an NVIDIA-internal Git remote.

## Audit Baseline

| Field | Value |
| --- | --- |
| Repository | `https://github.com/NVIDIA/cosmos-framework.git` |
| Revision | `ed8287fd7477113f8ac4f6b84290514d55cf0cdc` |
| Revision subject | `Release 2026-07-20: Cosmos3-Edge + Distillation support (#119)` |
| Dependency manager | `uv >= 0.11.3` |
| Python | `3.13` |
| Status | Audit baseline; not yet approved as the final release revision |

The revision is pinned in both `pyproject.toml` and `release-manifest.json`.
`uv.lock` records the resolved Git source and must be regenerated whenever the
revision changes.

## Internal Development History

The surgical development branch originally diverged from Cosmos Framework at
`90cd348877c37b888942c988b631eb1611bf2950`. That commit is used only to audit
the historical development delta. It is not a supported runtime dependency.

Surgical code is migrated according to these rules:

1. Prefer public framework APIs and declarative configuration.
2. Keep surgical datasets, preprocessing, model aliases, and examples in this
   repository.
3. Do not use runtime monkeypatches or overwrite files in `cosmos_framework`.
4. Do not publish DFW paths, internal object-store URLs, or internal Git URLs.
5. If a release capability requires an unavoidable framework modification,
   document the exact blocker before selecting a public fork or vendoring a
   minimal licensed implementation.

## Updating the Framework Revision

Framework updates require a dedicated pull request that:

1. Changes the full 40-character revision in `pyproject.toml`, this document,
   and `release-manifest.json`.
2. Regenerates `uv.lock` using the supported uv version.
3. Runs unit tests and clean-environment import tests.
4. Runs checkpoint load, export, T2V, I2V, single-control, and multicontrol GPU
   smoke tests when a release checkpoint is available.
5. Records any user-visible behavior change in the release notes.
