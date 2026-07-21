# Cosmos-H-Surgical

[![License](https://img.shields.io/badge/Code%20and%20Weights-OpenMDW--1.1-blue)](LICENSE.OpenMDW-1.1)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/nvidia/Cosmos-H-Surgical)
[![arXiv](https://img.shields.io/badge/arXiv-2512.23162-b31b1b)](https://arxiv.org/abs/2512.23162)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB)](https://www.python.org/)

Cosmos-H-Surgical is a surgical video world model built on NVIDIA Cosmos. This
branch prepares the Cosmos 3 codebase for the planned `v0.3.0` release.

> The Cosmos 3 checkpoint is still being selected and validated. The code on
> this branch must not be treated as a published model release. The complete
> Cosmos 2.5 release remains available on the
> [`cosmos-2.5`](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/cosmos-2.5)
> branch and under the signed
> [`v0.2.0`](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/v0.2.0)
> tag.

## Architecture

Cosmos-H-Surgical is a focused package layered on an immutable public revision
of [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework). The
framework is installed as a commit-pinned dependency and is not modified at
runtime.

```text
NVIDIA/cosmos-framework@ed8287fd7477113f8ac4f6b84290514d55cf0cdc
                              ^
                              |
Cosmos-H-Surgical 0.3.0 ------+
```

The current framework revision is an audit baseline. It becomes the final
release dependency only after checkpoint loading, inference, export, and
post-training compatibility tests pass. See [UPSTREAM.md](UPSTREAM.md).

## Repository Structure

```text
Cosmos-H-Surgical/
|-- cosmos_h_surgical/          # Surgical package and CLI
|-- examples/
|   |-- inference/              # Cosmos 3 input specifications
|   `-- post_training/          # Public post-training recipes
|-- tests/                      # Release and integration checks
|-- pyproject.toml              # Package and uv configuration
|-- uv.lock                     # Reproducible dependency lock
|-- UPSTREAM.md                 # Framework provenance and update policy
`-- release-manifest.json       # Public release artifact inventory
```

## Installation

The supported environment manager is [uv](https://docs.astral.sh/uv/). CUDA
13.0 is recommended and CUDA 12.8 is also supported. The selected group must
match the CUDA major version supported by the NVIDIA driver. An NVIDIA
Ampere-generation GPU or newer is required for model inference. Python 3.13
matches the pinned framework's own development environment and is provisioned
automatically by `uv`.

```bash
git clone https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical.git
cd Cosmos-H-Surgical

# CUDA 13, recommended
uv sync --group cu130

# CUDA 12.8
uv sync --group cu128
```

The root `uv.lock` and the framework commit in `pyproject.toml` are the source
of truth for the Python environment. `git-lfs` must be installed because the
pinned framework repository contains LFS-managed files. The commands below use
`uv run --no-sync` so `uv` preserves the CUDA group selected during installation.

## Inference

The package exposes the Cosmos Framework inference interface through a stable
project command:

```bash
uv run --no-sync cosmos-h-surgical infer \
  -i examples/inference/t2v.json \
  -o outputs/t2v \
  --checkpoint-path <COSMOS3_CHECKPOINT> \
  --seed 0
```

Until a `v0.3.0` checkpoint is published, pass an explicit compatible Cosmos 3
checkpoint path. The release will add a stable model alias only after the
checkpoint candidate and its hashes are approved.

Inspect the pinned framework dependency:

```bash
uv run --no-sync cosmos-h-surgical framework-info
```

Validate the release metadata before staging:

```bash
uv run --no-sync cosmos-h-surgical validate-release
```

## Post-Training

The selected CUDA environment includes both inference and post-training
dependencies. The project-owned training wrapper registers surgical experiments
with the pinned framework before the framework composes the TOML configuration.

```bash
uv run --no-sync torchrun --nproc_per_node=8 -m cosmos_h_surgical.training \
  --sft-toml examples/post_training/cosmos_h_surgical_vision_lora_480p.toml
```

The first migrated recipe covers 480P surgical T2V and I2V LoRA training. See
[examples/post_training/README.md](examples/post_training/README.md) for its
dataset contract and required environment variables. Mixed transfer/action
training remains a documented release gate rather than an unpublished API.

## Development

```bash
uv sync --group cu130 --group dev
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

The development branch must not contain DFW paths, internal object-store URLs,
credentials, or unpublished checkpoint paths. Tests enforce this boundary for
the release metadata and package sources.

## Release Plan

The planned release version is `v0.3.0`. The signed tag will be created only
after:

1. The final framework and model revisions are frozen.
2. All advertised inference modes pass clean-cache GPU smoke tests.
3. Checkpoint hashes match after a fresh Hugging Face download.
4. The model card, licenses, examples, and generated videos pass human review.

## License

Cosmos 3 code added in this release and the planned Cosmos 3 model weights are
provided under [OpenMDW-1.1](LICENSE.OpenMDW-1.1). Third-party software remains
subject to its own license terms. See [NOTICE](NOTICE) and
[ATTRIBUTIONS.md](ATTRIBUTIONS.md).

## Citation

```bibtex
@misc{he2026cosmoshsurgicallearningsurgicalrobot,
  title={Cosmos-H-Surgical: Learning Surgical Robot Policies from Videos via World Modeling},
  author={Yufan He and Pengfei Guo and Mengya Xu and Zhaoshuo Li and Andriy Myronenko and Dillan Imans and Bingjie Liu and Dongren Yang and Mingxue Gu and Yongnan Ji and Yueming Jin and Ren Zhao and Baiyong Shen and Daguang Xu},
  year={2026},
  eprint={2512.23162},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2512.23162},
}
```
