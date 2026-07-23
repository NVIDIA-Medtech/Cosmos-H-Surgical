# Cosmos-H-Surgical

[![License](https://img.shields.io/badge/Code%20and%20Weights-OpenMDW--1.1-blue)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/nvidia/Cosmos-H-Surgical)
[![arXiv](https://img.shields.io/badge/arXiv-2512.23162-b31b1b)](https://arxiv.org/abs/2512.23162)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB)](https://www.python.org/)

A surgical video world foundation model suite based on [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos) and [SurgWorld](https://arxiv.org/abs/2512.23162), part of the NVIDIA MedTech Open Models.

<p align="center">
<img width="935" height="224" alt="Cosmos-H-Surgical overview" src="https://github.com/user-attachments/assets/69656a2a-e6b0-4ca9-aa6c-5c2f572a2656" />
</p>

## Overview

Cosmos-H-Surgical delivers high-quality video prediction and transfer for surgical scenes, including future-state simulation and control-conditioned generation across modalities. The Cosmos 3 release unifies Predict and Transfer in a single mixed-capability checkpoint that supports both image-to-video prediction and multi-modal control-based generation. This project was conducted by NVIDIA in collaboration with [Chinese University of Hong Kong](https://www.cse.cuhk.edu.hk/~qdou/), [National University of Singapore](https://yuemingjin.github.io/), and [Shanghai Jiao Tong University](https://gc.sjtu.edu.cn/about/faculty-staff/faculty-directory/faculty-detail/75745/).

## News

- **[July 2026]** — Released Cosmos-H-Surgical, built on NVIDIA Cosmos 3, with **Predict** and **Transfer** unified in a single mixed-capability checkpoint.
- **[July 2026]** — Added OpenMDW-1.1 checkpoint editions for Predict and Transfer while retaining the original checkpoint paths and license terms.
- **[March 2026]** — Released [SurgΣ](https://arxiv.org/abs/2603.16822): a large-scale multimodal surgical dataset and foundation model suite for surgical intelligence.
- **[March 2026]** — Released [BSA](https://arxiv.org/abs/2603.12787): generalized recognition of basic surgical actions enabling skill assessment and VLM-based surgical planning.
- **[March 2026]** — Released [Cosmos-H-Surgical-Predict](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/cosmos-2.5/predict) and [Cosmos-H-Surgical-Transfer](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/cosmos-2.5/transfer) as part of the NVIDIA MedTech Open Models.


> The complete Cosmos 2.5 release remains available on the
> [`cosmos-2.5`](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/cosmos-2.5)
> branch and under the signed
> [`v0.2.0`](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/v0.2.0)
> tag.

## Repository Structure

```text
Cosmos-H-Surgical/
|-- cosmos_h_surgical/          # Surgical package and CLI
|-- datasets/                   # Synthetic toy post-training dataset
|-- examples/
|   |-- post_training/          # Public post-training recipes
|-- inputs/                     # Predict, Transfer, and prompt examples
|-- docs/                       # Setup, inference, and migration guides
|-- tests/                      # Release and integration checks
|-- pyproject.toml              # Package and uv configuration
|-- uv.lock                     # Reproducible dependency lock
|-- UPSTREAM.md                 # Framework provenance and update policy
`-- release-manifest.json       # Public release artifact inventory
```
### Scaling World State Diversity Examples

<p align="center">
  <video src="https://github.com/user-attachments/assets/90f67345-0e42-4c53-a328-4dde0821783f"
         width="100%"
         alt="Input video"
         controls>
  </video>
</p>

## Architecture

Cosmos-H-Surgical is a focused package layered on an immutable public revision
of [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework). The
framework is installed as a commit-pinned dependency. See [UPSTREAM.md](UPSTREAM.md).


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


## License

Code and model weights in this release are
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

@misc{zeng2026surgsigma,
  title={Surg$\Sigma$: A Spectrum of Large-Scale Multimodal Data and Foundation Models for Surgical Intelligence},
  author={Zhitao Zeng and Mengya Xu and Jian Jiang and Pengfei Guo and Yunqiu Xu and Zhu Zhuo and Chang Han Low and Yufan He and Dong Yang and Chenxi Lin and Yiming Gu and Jiaxin Guo and Yutong Ban and Daguang Xu and Qi Dou and Yueming Jin},
  year={2026},
  eprint={2603.16822},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2603.16822},
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
