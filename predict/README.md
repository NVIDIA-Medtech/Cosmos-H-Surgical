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

[**Cosmos-H-Surgical-Predict**](docs/inference.md): Base [2B checkpoints](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/tree/main/base), trained from the ground up for Physical AI and robotics for surgical tasks.


| Model Name | Capability | Input |
| --- | --- | --- |
| [**Cosmos-H-Surgical-Predict**](docs/inference.md) | Base Model | text + image  |

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

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
