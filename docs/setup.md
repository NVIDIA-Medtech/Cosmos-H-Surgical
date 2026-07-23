# Setup

Cosmos-H-Surgical is a thin package over a commit-pinned revision of
[NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework). The
supported environment manager is [uv](https://docs.astral.sh/uv/).

The Cosmos 3 release candidate is still under validation. Until the v0.3.0
checkpoint is published, inference requires an explicit local checkpoint path.

## Requirements

- Linux on `x86_64` or `aarch64`
- Python 3.13, provisioned by uv
- An NVIDIA Ampere-generation GPU or newer
- A driver compatible with CUDA 12.8 or CUDA 13.0
- Git and Git LFS
- uv 0.11.3 or newer

The release candidate has been validated with eight H100 GPUs. Smaller GPU
configurations may require different parallelism or offloading settings and are
not yet part of the release test matrix.

## Install

Clone the repository and initialize Git LFS:

```bash
git clone https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical.git
cd Cosmos-H-Surgical
git lfs install
git lfs pull
```

Select exactly one CUDA dependency group. CUDA 13 is recommended:

```bash
uv sync --group cu130
```

For CUDA 12.8:

```bash
uv sync --group cu128
```

Run exactly one sync command, then activate the project environment:

```bash
source .venv/bin/activate
```

The two CUDA groups conflict intentionally. Do not install both into the same
environment. The root `uv.lock` and the full Cosmos Framework Git revision in
`pyproject.toml` define the reproducible environment.

Commands in this repository assume `.venv` is active. Activating it preserves
the CUDA group selected above without repeating dependency-group options on
every command.

## Verify

Confirm the package and immutable framework provenance:

```bash
cosmos-h-surgical --version
cosmos-h-surgical framework-info
```

The reported repository and revision must match [UPSTREAM.md](../UPSTREAM.md).
Inspect the available inference arguments with:

```bash
cosmos-h-surgical infer --help
```

## Development Environment

Add the development dependency group to the selected CUDA environment:

```bash
uv sync --group cu130 --group dev
source .venv/bin/activate
pytest
ruff check .
ruff format --check .
```

Replace `cu130` with `cu128` when testing the CUDA 12.8 environment.

## Storage and Caches

Model snapshots, tokenizers, and auxiliary checkpoints can require substantial
disk space. Set `HF_HOME` before installation or inference when the default
cache location is unsuitable:

```bash
export HF_HOME=/path/to/huggingface-cache
```

Inference outputs are written below the directory passed with `--output-dir`
and are ignored by Git. See [environment_variables.md](environment_variables.md)
for the complete project-specific environment-variable reference.

For general framework installation details, consult the
[setup guide at the pinned framework revision](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/setup.md).
