# Cosmos-H-Surgical

[![License](https://img.shields.io/badge/Code-Apache%202.0-blue)](LICENSE)
[![Weights License](https://img.shields.io/badge/Weights-NVIDIA--OneWay--Noncommercial--License-green)](LICENSE.weights)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/nvidia/Cosmos-H-Surgical)
[![arXiv](https://img.shields.io/badge/arXiv-2512.23162-b31b1b)](https://arxiv.org/abs/2512.23162)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)

A surgical video world foundation model suite based on [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) and [SurgWorld](https://arxiv.org/abs/2512.23162), part of the NVIDIA MedTech Open Models.

<p align="center">
<img width="935" height="224" alt="Cosmos-H-Surgical overview" src="https://github.com/user-attachments/assets/69656a2a-e6b0-4ca9-aa6c-5c2f572a2656" />
</p>

## Overview

Cosmos-H-Surgical delivers high-quality video prediction and transfer for surgical scenes, including future-state simulation and control-conditioned generation across modalities. It comprises two sub-projects — **Predict** for image-to-video generation and **Transfer** for multi-modal control-based generation — both adapted from NVIDIA Cosmos 2.5 for surgical video data. For action-conditioned surgical simulation with robot kinematics, see the companion repo [Cosmos-H-Surgical-Simulator](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator).

This project was conducted by NVIDIA in collaboration with [Chinese University of Hong Kong](https://www.cse.cuhk.edu.hk/~qdou/), [National University of Singapore](https://yuemingjin.github.io/), and [Shanghai Jiao Tong University](https://gc.sjtu.edu.cn/about/faculty-staff/faculty-directory/faculty-detail/75745/).

## News

- **[March 2026]** — Released [Cosmos-H-Surgical-Predict](predict/) and [Cosmos-H-Surgical-Transfer](transfer/) as part of the NVIDIA MedTech Open Models.

## Model Variants

| Model | Base Model | Params | Capability | Input | HuggingFace | License |
|-------|-----------|--------|------------|-------|-------------|---------|
| [Cosmos-H-Surgical-Predict](predict/) | [Cosmos-Predict2.5-2B](https://github.com/nvidia-cosmos/cosmos-predict2.5) | 2B | Future-state video prediction (Image2World) | text + image | [Weights](https://huggingface.co/nvidia/Cosmos-H-Surgical) | [NVIDIA-OneWay-Noncommercial-License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |
| [Cosmos-H-Surgical-Transfer](transfer/) | [Cosmos-Transfer2.5-2B](https://github.com/nvidia-cosmos/cosmos-transfer2.5) | 2B | Control-conditioned generation (depth, edge, seg, blur) | text + video + control maps | [Weights](https://huggingface.co/nvidia/Cosmos-H-Surgical) | [NVIDIA-OneWay-Noncommercial-License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |

## Repository Structure

```
Cosmos-H-Surgical/
├── predict/                          # Cosmos-H-Surgical-Predict
│   ├── cosmos_predict2/              # Predict package (experiments, datasets)
│   ├── packages/                     # Workspace deps (cosmos-oss, cosmos-cuda, cosmos-gradio)
│   ├── docs/                         # Setup, inference, post-training
│   ├── examples/                     # Inference scripts
│   ├── assets/                       # Example inputs (JSON configs, images)
│   ├── Dockerfile
│   └── pyproject.toml
├── transfer/                         # Cosmos-H-Surgical-Transfer
│   ├── cosmos_transfer2/             # Transfer package
│   ├── packages/                     # Workspace deps
│   ├── docs/                         # Setup, inference, post-training
│   ├── examples/                     # Inference scripts
│   ├── assets/                       # Example inputs (JSON specs, control maps)
│   ├── Dockerfile
│   └── pyproject.toml
├── LICENSE                           # Apache 2.0 (source code)
└── LICENSE.weights                   # NVIDIA-OneWay-Noncommercial-License License (weights)
```

## Quick Start

**System Requirements**: NVIDIA Ampere+ GPU (A100, H100, B200), Linux x86-64, CUDA 12.8+, Python 3.10+

### Install

```bash
git clone git@github.com:NVIDIA-Medtech/Cosmos-H-Surgical.git
cd Cosmos-H-Surgical
git lfs pull
```

Each sub-project has its own environment. For example, to set up **Predict**:

```bash
cd predict
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install && uv sync --extra=cu128
source .venv/bin/activate
```

See [predict/docs/setup.md](predict/docs/setup.md) or [transfer/docs/setup.md](transfer/docs/setup.md) for full instructions including Docker.

### Inference

**Predict** (Image2World):
```bash
cd predict
python examples/inference.py -i assets/base/coagulation.json -o outputs/base_video2world --inference-type=video2world
```

**Transfer** (control-conditioned):
```bash
cd transfer
python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth
```

## Documentation

| Topic | Predict | Transfer |
|-------|---------|----------|
| Setup | [predict/docs/setup.md](predict/docs/setup.md) | [transfer/docs/setup.md](transfer/docs/setup.md) |
| Inference | [predict/docs/inference.md](predict/docs/inference.md) | [transfer/docs/inference.md](transfer/docs/inference.md) |
| Post-training | [predict/docs/post-training.md](predict/docs/post-training.md) | [transfer/docs/post-training.md](transfer/docs/post-training.md) |
| Troubleshooting | [predict/docs/troubleshooting.md](predict/docs/troubleshooting.md) | [transfer/docs/troubleshooting.md](transfer/docs/troubleshooting.md) |

## Performance

### Transfer Inference (Segmentation control, 720p 16FPS, 93 frames, 65.4 GB VRAM)

| GPU Hardware | Generation Time | End-to-End Time |
|-------------|----------------|-----------------|
| NVIDIA B200 | 92.25 sec | 186.92 sec |
| NVIDIA H100 NVL | 445.52 sec | 895.33 sec |
| NVIDIA H100 PCIe | 264.13 sec | 533.58 sec |
| NVIDIA H20 | 683.65 sec | 1370.39 sec |

End-to-end time measured for 121-frame input video (two 93-frame chunk generations). Guardrails disabled.

## License

| Component | License |
|-----------|---------|
| Source code | [Apache 2.0](LICENSE) |
| Cosmos-H-Surgical-Predict weights | [NVIDIA-OneWay-Noncommercial-License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |
| Cosmos-H-Surgical-Transfer weights | [NVIDIA-OneWay-Noncommercial-License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) |

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

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

@misc{xu2026generalizedrecognitionbasicsurgicalactions,
  title={Generalized Recognition of Basic Surgical Actions Enables Skill Assessment and Vision-Language-Model-based Surgical Planning},
  author={Mengya Xu and Daiyun Shen and Jie Zhang and Hon Chi Yip and Yujia Gao and Cheng Chen and Dillan Imans and Yonghao Long and Yiru Ye and Yixiao Liu and Rongyun Mai and Kai Chen and Hongliang Ren and Yutong Ban and Guangsuo Wang and Francis Wong and Chi-Fai Ng and Kee Yuan Ngiam and Russell H. Taylor and Daguang Xu and Yueming Jin and Qi Dou},
  year={2026},
  eprint={2603.12787},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.12787},
}
```

## Resources

- [Cosmos-H-Surgical Paper (arXiv)](https://arxiv.org/abs/2512.23162)
- [Basic Surgical Actions Dataset Paper (arXiv)](https://arxiv.org/abs/2603.12787)
- [HuggingFace Collection](https://huggingface.co/nvidia/Cosmos-H-Surgical)
- [NVIDIA Cosmos Platform](https://www.nvidia.com/en-us/ai/cosmos/)
- [Cosmos-H-Surgical-Simulator](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical-Simulator) — Sister repo (action-conditioned surgical simulation)
- [NVIDIA MedTech Open Models](https://github.com/NVIDIA-Medtech)
