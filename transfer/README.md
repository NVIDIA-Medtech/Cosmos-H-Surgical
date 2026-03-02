<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  <a href="https://www.nvidia.com/en-us/ai/cosmos/"> Product Website</a>&nbsp | 🤗 <a href="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B">Hugging Face</a>&nbsp | <a href="https://arxiv.org/abs/2511.00062">Paper</a> | <a href="https://research.nvidia.com/labs/dir/cosmos-transfer2.5/">Paper Website</a> | <a href="https://github.com/nvidia-cosmos/cosmos-cookbook">Cosmos Cookbook</a>
</p>

NVIDIA Cosmos™ is a platform purpose-built for physical AI, featuring state-of-the-art generative world foundation models (WFMs), robust guardrails, and an accelerated data processing and curation pipeline. Designed specifically for real-world systems, Cosmos enables developers to rapidly advance physical AI applications such as robotic surgery, AV, robots, and video analytics AI agents.

## News
* [March 16, 2026] As part of the NVIDIA Clara Open Models family, we released [Cosmos-H-Surgical-Transfer](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical)

## Cosmos-H-Surgical-Transfer

**Cosmos-H-Surgical-Transfer** is a multi-ControlNet framework designed to accept structured inputs from multiple video modalities, including RGB, depth, segmentation, and more. Users can configure generation through JSON-based controlnet_specs and run inference with just a few commands. It supports single-video inference, automatic control map generation, and multi-GPU setups. Cosmos-H-Surgical-Transfer is built upon the [Cosmos-Transfer2.5-2B](https://github.com/nvidia-cosmos/cosmos-transfer2.5) model and is adapted specifically for surgical video data.

Physical AI trains upon data generated in two important data augmentation workflows.

### Simulation 2 Real Augmentation

**Input prompt:**
> Real surgery scene: right instrument coagulates the cystic artery.

<table>
  <tr>
    <th>Computed Control</th>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/64c49407-f0ca-49a9-80ae-2a5ad9ae7102" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/c7b4cb74-2e6d-4ad2-aa13-95b91c5ef05a" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/1017d1c6-01f0-4ea9-89ca-7c9affa61b67" width="100%" alt="Control map video" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/6d1bb0b2-154d-4a28-9b67-aaf245b6c261" width="100%" alt="Output video" controls></video>
    </td>
  </tr>
</table>

### Real 2 Real Augmentation (With Guided Generation)

**Input prompt:**

> Real surgery scene: Use the robotic forceps to puncture a needle into a soft tissue structure.
<table>
  <tr>
    <th>Input Video</th>
    <th>Computed Control</th>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/2c4f099f-843c-4caa-b54d-e78c8887fd13" width="100%" alt="Input video" controls></video>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/3ff59c54-978b-4ebc-aa74-5cc076d56b5c" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/b3cc35d1-911b-45e5-903c-7b76376aeffe" width="100%" alt="Guided Generation Mask" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/70044c14-39e6-4316-b30b-45baee4d3f96" width="100%" alt="Output video" controls></video>
    </td>
  </tr>
</table>

### Scaling World State Diversity Examples

<video src="https://github.com/user-attachments/assets/9ba56017-5dfb-452f-bf71-320157225656" width="100%" alt="Input video" controls></video>

## Cosmos-Transfer2.5 Model Family

Cosmos-Transfer supports data generation in multiple industry verticals, outlined below. Please check back as we continue to add more specialized models to the Transfer family!

[**Cosmos-H-Surgical-Transfer**](docs/inference.md): General [checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B), trained from the ground up for Physical AI and robotics for surgical tasks.

## User Guide

* [Setup Guide](docs/setup.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Inference](docs/inference.md)
  * [Inference with Guided Generation](docs/inference_guided_generation.md)
* [Post-training](docs/post-training.md)
  * [Single View](docs/post-training_singleview.md)

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
