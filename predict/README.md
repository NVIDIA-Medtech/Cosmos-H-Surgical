<p align="center">
  🤗 <a href="https://huggingface.co/nvidia/Cosmos-H-Surgical">Hugging Face</a>&nbsp | <a href="https://arxiv.org/abs/2512.23162">Paper</a>&nbsp | <a href="https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical">Repository</a>
</p>

Cosmos-H-Surgical-Predict is a surgical video world foundation model for simulating and predicting future surgical states, part of the [Cosmos-H-Surgical](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical) suite and the NVIDIA MedTech Open Models.

## News
* [March 16, 2026] As part of the NVIDIA MedTech Open Models, we released [Cosmos-H-Surgical-Predict](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/main/predict)

## Cosmos-H-Surgical-Predict

We introduce Cosmos-H-Surgical-Predict, specialized for simulating and predicting the future state of the world in the form of video. Cosmos-H-Surgical-Predict is a flow based model utilizes Cosmos-Reason1, a Physical AI reasoning vision language model (VLM), as the text encoder.  Cosmos-H-Surgical-Predict is built upon the [Cosmos-Predict2.5-2B](https://github.com/nvidia-cosmos/cosmos-predict2.5) model and is adapted specifically for surgical video data.

### Image2World

<details><summary>Input prompt</summary>
real surgery scene: right instrument coagulates the cystic mesentery while  left instrument retracts the cystic mesentery.
</details>

| Input image | Output video
| --- | --- |
| <img src="https://github.com/user-attachments/assets/bb6a9992-7d25-451b-8a9a-219bb9cb36e0" width="495" alt="Input image" > | <video src="https://github.com/user-attachments/assets/5bbf8430-638e-4679-a4e1-fc8cc0d6a599" width="500" alt="Output video" controls></video> |

<details><summary>Input prompt</summary>
real surgery scene: left needle driver passes needle to right needle driver.
</details>

| Input Video | Output Video
| --- | --- |
| <img src="https://github.com/user-attachments/assets/cb2029f8-f934-4049-ba31-2f2be6622835" width="495" alt="Input image" > | <video src="https://github.com/user-attachments/assets/113a1928-6bd7-4caa-843c-da940a8d7c06" width="500" alt="Output video" controls></video> |

### Scaling World State Diversity Examples

<video src="https://github.com/user-attachments/assets/e6ab161c-0608-4521-b42c-339e0cc47baf" width="100%" alt="Diverse videos" controls></video>

## Cosmos-H-Surgical-Predict Model Family

Our world simulation models, Cosmos-Predict's fundamental capability is predicting future world states in video form supporting multimodal inputs. We have open sourced both pre-trained foundation models as well as post-trained models accelerating multiple domains. Please check back as we continue to add more specialized models and capabilities to the Predict family!

[**Cosmos-H-Surgical-Predict**](docs/inference.md): 2B checkpoints adapted for Physical AI and surgical robotics tasks.


| Model Name | Model Key | Capability | Input | License |
| --- | --- | --- | --- | --- |
| [**Cosmos-H-Surgical-Predict**](docs/inference.md) (legacy) | `2B/post-trained` | Base Model | text + image | [NVIDIA-OneWay-Noncommercial-License](../LICENSE.weights) |
| [**Cosmos-H-Surgical-Predict**](docs/inference.md) (OpenMDW) | `2B/post-trained/openmdw-1.1` | Base Model | text + image | [OpenMDW-1.1](../LICENSE.OpenMDW-1.1) |

## User Guide

* [Setup Guide](docs/setup.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Inference](docs/inference.md)
* [Post-Training](docs/post-training.md)
  * [Image2World Cosmos-H-Surgical-Assets](docs/post-training_cosmos_h_surgical_assets.md)
  * [Image2World Cosmos-H-Surgical-Assets LoRA](docs/post-training_cosmos_h_surgical_assets_lora.md)

## Contributing

We welcome contributions. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

Cosmos-H-Surgical source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

Legacy Cosmos-H-Surgical-Predict checkpoints are released under [LICENSE.weights](../LICENSE.weights). Checkpoints under `openmdw-1.1/predict/` are released under [OpenMDW-1.1](../LICENSE.OpenMDW-1.1). See [NOTICE.weights](../NOTICE.weights) and [release-manifest.json](../release-manifest.json) for the exact license scope and artifact checksums.
