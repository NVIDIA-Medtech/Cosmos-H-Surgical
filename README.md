# Cosmos-H-Surgical

Surgical video world foundation models built on [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/), part of the NVIDIA Clara Open Models family. This repository contains two independent sub-projects:

- **`predict/`** — **[Cosmos-H-Surgical-Predict](predict/README.md)**  
  Simulates and predicts future world states as video (Image2World). Flow-based model using Cosmos-Reason1 as the text encoder, built on [Cosmos-Predict2.5-2B](https://github.com/nvidia-cosmos/cosmos-predict2.5) and adapted for surgical video.

- **`transfer/`** — **[Cosmos-H-Surgical-Transfer](transfer/README.md)**  
  High-quality world simulation conditioned on spatial control inputs. Multi-ControlNet framework for RGB, depth, segmentation, and other modalities via JSON `controlnet_specs`. Built on [Cosmos-Transfer2.5-2B](https://github.com/nvidia-cosmos/cosmos-transfer2.5) and adapted for surgical video (Simulation→Real and Real→Real augmentation).

Each sub-project has its own virtual environment, dependencies, and docs. Use the links above for capabilities, examples, and model info.

## Project structure

```
Cosmos-H-Surgical-gitlab/
├── predict/                          # Cosmos-H-Surgical-Predict
│   ├── cosmos_predict2/               # Predict package (experiments, datasets, etc.)
│   ├── packages/                      # Workspace dependencies
│   │   ├── cosmos-oss/
│   │   ├── cosmos-cuda/
│   │   └── cosmos-gradio/
│   ├── docs/                          # Setup, inference, post-training
│   ├── pyproject.toml
│   └── README.md
├── transfer/                          # Cosmos-H-Surgical-Transfer
│   ├── cosmos_transfer2/              # Transfer package
│   ├── packages/
│   │   ├── cosmos-oss/
│   │   ├── cosmos-cuda/
│   │   └── cosmos-gradio/
│   ├── docs/                          # Setup, inference, post-training
│   ├── pyproject.toml
│   └── README.md
└── README.md
```

## Predict

See [predict/docs/setup.md](predict/docs/setup.md) for system deps, Docker, and checkpoint download.

## Transfer

See [transfer/docs/setup.md](transfer/docs/setup.md) for system deps, Docker, and checkpoint download.

## Usage

- **Predict:** [predict/README.md](predict/README.md) and [predict/docs/inference.md](predict/docs/inference.md) — Image2World, examples, and post-training (including LoRA).  
  Package: `cosmos_predict2`.

- **Transfer:** [transfer/README.md](transfer/README.md) and [transfer/docs/inference.md](transfer/docs/inference.md) — control-based generation, guided generation, and post-training (e.g. single view).  
  Package: `cosmos_transfer2`.

## User guides (per sub-project)

| Topic            | Predict | Transfer |
|------------------|---------|----------|
| Setup            | [predict/docs/setup.md](predict/docs/setup.md) | [transfer/docs/setup.md](transfer/docs/setup.md) |
| Inference        | [predict/docs/inference.md](predict/docs/inference.md) | [transfer/docs/inference.md](transfer/docs/inference.md) |
| Post-training    | [predict/docs/post-training.md](predict/docs/post-training.md) | [transfer/docs/post-training.md](transfer/docs/post-training.md) |
| Troubleshooting  | [predict/docs/troubleshooting.md](predict/docs/troubleshooting.md) | [transfer/docs/troubleshooting.md](transfer/docs/troubleshooting.md) |
