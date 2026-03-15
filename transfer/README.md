<p align="center">
  🤗 <a href="https://huggingface.co/nvidia/Cosmos-H-Surgical">Hugging Face</a>&nbsp | <a href="https://arxiv.org/abs/2512.23162">Paper</a>&nbsp | <a href="https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical">Repository</a>
</p>

Cosmos-H-Surgical-Transfer is a multi-ControlNet framework for control-conditioned surgical video generation, part of the [Cosmos-H-Surgical](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical) suite and the NVIDIA MedTech Open Models.

## News
* [March 16, 2026] As part of the NVIDIA MedTech Open Models, we released [Cosmos-H-Surgical-Transfer](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/main/transfer)

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
      <video src="https://github.com/user-attachments/assets/0e69bf77-cb8f-48ae-9022-3200fedf7588" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/35817783-5a61-4302-a38d-fcd7c32944d9" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/253afe1b-55ad-417c-9a4f-301dc1fb852c" width="100%" alt="Control map video" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/a712073a-40d3-4ed7-884c-416695ec4f55" width="100%" alt="Output video" controls></video>
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
      <video src="https://github.com/user-attachments/assets/fbc14bfc-d465-42d0-960a-d02664ec097d" width="100%" alt="Input video" controls></video>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/c66e7533-0d7f-47d9-a9fe-3fa34f58b1c8" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/658c60bd-6307-45ca-a47a-4d8096608d9e" width="100%" alt="Guided Generation Mask" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/c707ec7e-4e32-48cd-97fa-e970636a78d1" width="100%" alt="Output video" controls></video>
    </td>
  </tr>
</table>

### Scaling World State Diversity Examples

<video src="https://github.com/user-attachments/assets/7022851f-18d4-4565-9739-5c34dd9ea480" width="100%" alt="Input video" controls></video>

## Cosmos-H-Surgical-Transfer Model Family

[**Cosmos-H-Surgical-Transfer**](docs/inference.md): General [checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B), trained from the ground up for Physical AI and robotics for surgical tasks.

## User Guide

* [Setup Guide](docs/setup.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Inference](docs/inference.md)
  * [Inference with Guided Generation](docs/inference_guided_generation.md)
* [Post-training](docs/post-training.md)
  * [Single View](docs/post-training_singleview.md)

## Contributing

We welcome contributions. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

Cosmos-H-Surgical source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

Cosmos-H-Surgical models are released under the [NVIDIA License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyIsIm5jaWQiOiJzby15b3V0LTg3MTcwMS12dDQ4In0=). You are responsible for ensuring that your use of NVIDIA AI Foundation Models complies with all applicable laws.
