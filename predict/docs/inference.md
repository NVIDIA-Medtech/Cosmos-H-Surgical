# Inference Guide

This guide provides instructions on running inference with Cosmos-H-Surgical-Predict models.

<p align="center">
  <img width="500" alt="cosmos-predict-diagram" src="https://github.com/user-attachments/assets/8f436cdd-3d04-46ea-b333-d8e9ccdc6d9c">
</p>

## Prerequisites

1. [Setup Guide](setup.md)

## Example

Run inference with example asset:

```bash
python examples/inference.py -i assets/base/coagulation.json -o outputs/base_video2world --inference-type=video2world
```

To enable multi-GPU inference with 8 GPUs, use [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):

```bash
torchrun --nproc_per_node=8 examples/inference.py -i assets/base/coagulation.json -o outputs/base_video2world --inference-type=video2world
```

To generate longer videos with autoregressive sliding window mode:

```bash
python examples/inference.py -i assets/base/aspiration_long.json -o outputs/autoregressive
```

To run all example assets:

```shell
torchrun --nproc_per_node=8 examples/inference.py -i assets/base/*.json -o outputs/base
```

To see all available options:

```bash
python examples/inference.py --help
```

Parameters are specified as json:

```jsonc
{
  // Inference type: image2world
  "inference_type": "image2world",
  // Sample name
  "name": "coagulation",
  // Input prompt
  "prompt": "real surgery scene: right instrument coagulates the cystic mesentery while left instrument retracts the cystic mesentery ...",
  // Path to the input image file
  "input_path": "coagulation.jpg"
}
```

### Outputs

#### image2world/Aspiration

<video src="https://github.com/user-attachments/assets/6b9f7151-ac20-4405-81e1-d05d78cbf65a" width="500" alt="text2world/snowy_stop_light" controls></video>

#### image2world/Coagulation

<video src="https://github.com/user-attachments/assets/5bbf8430-638e-4679-a4e1-fc8cc0d6a599" width="500" alt="image2world/robot_welding" controls></video>

#### image2world/Needle Grasping

<video src="https://github.com/user-attachments/assets/113a1928-6bd7-4caa-843c-da940a8d7c06" width="500" alt="video2world/sand_mining" controls></video>

## Tips

### Multi-GPU

Context parallelism distributes inference across multiple GPUs, with each GPU generating a subset of the video frames.

* The number of GPUs should ideally be a divisor of the number of frames in the generated video.
* All GPUs should have the same model capacity and memory.
* Context parallelism works best with the 14B model where memory constraints are significant.
* Requires NCCL support and proper GPU interconnect for efficient communication.
* Significant speedup for video generation while maintaining the same quality.

### Prompt Engineering

For best results with Cosmos models, create detailed prompts that emphasize physical realism, natural laws, and real-world behaviors. Describe specific objects, materials, lighting conditions, and spatial relationships while maintaining logical consistency throughout the scene.

Incorporate photography terminology like composition, lighting setups, and camera settings. Use concrete terms like "natural lighting" or "wide-angle lens" rather than abstract descriptions, unless intentionally aiming for surrealism. Include negative prompts to explicitly specify undesired elements.

The more grounded a prompt is in real-world physics and natural phenomena, the more physically plausible and realistic the generated image will be.
