# Cosmos-H-Surgical

[![License](https://img.shields.io/badge/Code%20and%20Weights-OpenMDW--1.1-blue)](LICENSE)
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
framework is installed as a commit-pinned dependency. Its installed source
files are not overwritten.

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
|-- docs/                       # Setup, inference, and migration guides
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

Run exactly one of the two sync commands, then activate the selected
environment for the rest of the session:

```bash
source .venv/bin/activate
```

The root `uv.lock` and the framework commit in `pyproject.toml` are the source
of truth for the Python environment. `git-lfs` must be installed because the
pinned framework repository contains LFS-managed files. Commands below assume
the environment is active. See [docs/setup.md](docs/setup.md) for requirements
and environment validation.

## Inference

The v0.3.0 release focuses on surgical I2V prediction and edge, depth,
segmentation, and blur transfer. Inference requires a structured JSON prompt;
plain natural-language prompts should first be converted with the prompt
upsampler. The public `Cosmos-H-Surgical` checkpoint is selected by default.

```bash
torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical infer \
  -i inputs/predict/surgical_predict_smoke.json \
  --output-dir outputs/i2v \
  --seed 0
```

Pass `--checkpoint-path /path/to/checkpoint` only when testing an explicit
local export. Private release candidates can override the Hugging Face
repository and revision through the variables documented in
[docs/environment_variables.md](docs/environment_variables.md).

The [prepared input bundle](inputs/README.md) also includes a nine-action
prediction manifest and two validation examples for every transfer control.
See [docs/inference.md](docs/inference.md) for the structured input format,
checkpoint contract, I2V, and transfer commands. See
[docs/prompt_upsampling.md](docs/prompt_upsampling.md) for converting short
surgical descriptions into structured prompts.

Inspect the pinned framework dependency:

```bash
cosmos-h-surgical framework-info
```

## Documentation

| Guide | Description |
| --- | --- |
| [Setup](docs/setup.md) | CUDA 13/12.8 installation and environment verification. |
| [Inference](docs/inference.md) | Structured prompts, checkpoints, I2V, and all transfer controls. |
| [Prompt upsampling](docs/prompt_upsampling.md) | Convert short surgical prompts into Cosmos 3 JSON prompts. |
| [Environment variables](docs/environment_variables.md) | Inference, prompt-upsampling, and training variables. |
| [Troubleshooting](docs/troubleshooting.md) | Installation, checkpoint, input, and distributed failures. |
| [Code structure](docs/code_structure.md) | Package architecture and framework ownership boundary. |
| [Cosmos 2.5 migration](docs/migration_from_cosmos25.md) | Archive locations, command mapping, and compatibility. |
| [Post-training](docs/post_training.md) | Predict and Transfer LoRA recipes and dataset contracts. |

Validate the release metadata before staging:

```bash
cosmos-h-surgical validate-release
```

## Post-Training

The selected CUDA environment includes both inference and post-training
dependencies. The project-owned training wrapper registers surgical experiments
with the pinned framework before the framework composes the TOML configuration.

```bash
torchrun --nproc_per_node=8 -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_predict_lora_480p.toml
```

Separate recipes cover 480P surgical Predict and Transfer LoRA training. See
[docs/post_training.md](docs/post_training.md) for the development-format
manifest, caption and control-sidecar contracts, preparation and validation
commands, and required environment variables. Action training is not part of
the v0.3.0 public interface.

## Development

```bash
uv sync --group cu130 --group dev
source .venv/bin/activate
pytest
ruff check .
ruff format --check .
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
provided under [OpenMDW-1.1](LICENSE). Third-party software remains
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
