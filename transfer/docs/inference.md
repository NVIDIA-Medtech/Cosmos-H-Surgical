# Cosmos-H-Surgical-Transfer: World Generation with Adaptive Multimodal Control
This guide provides instructions on running inference with Cosmos-H-Surgical-Transfer models.

![Architecture](../assets/Cosmos-Transfer2-2B-Arch.png)

### Pre-requisites
1. Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

### Hardware Requirements

The following table shows the GPU memory requirements for different Cosmos-H-Surgical-Transfer models for single-GPU inference:

| Model | Required GPU VRAM |
|-------|-------------------|
| Cosmos-H-Surgical-Transfer | 65.4 GB |

### Inference performance

#### Segmentation
The table below shows generation times(*) across different NVIDIA GPU hardware for single-GPU inference:

| GPU Hardware | Cosmos-H-Surgical-Transfer 93 frame generation time | Cosmos-H-Surgical-Transfer E2E time (**)|
|--------------|---------------|---------------|
| NVIDIA B200 | 92.25 sec | 186.92 |
| NVIDIA H100 NVL | 445.52 sec | 895.33 |
| NVIDIA H100 PCIe | 264.13 sec | 533.58 |
| NVIDIA H20 | 683.65 sec | 1370.39 |

\* Generation times are listed for 720P video with 16FPS with segmentation control input and disabled guardrails. \
\** E2E time is measured for input video with 121 frames, which results in two 93 frame "chunk" generations.

## Inference with Pre-trained Cosmos-H-Surgical-Transfer Models

**For more detailed guidance about the control modalities and examples, checkout our Cosmos Cookbook [Control-Modalities](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/control_modalities/overview.html) recipe.**

Individual control variants can be run on a single GPU:
```bash
python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth
```

The legacy checkpoints remain the default. To use an OpenMDW-1.1 checkpoint, append the edition to the model key:

```bash
python examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json \
  -o outputs/openmdw_depth --model=depth/openmdw-1.1
```

For multi-control inference, use the explicit `multicontrol/openmdw-1.1` model key. This loads the depth, edge, segmentation, and visibility checkpoints from the OpenMDW-1.1 edition:

```bash
python examples/inference.py \
  -i assets/coagulation_example/multicontrol/coagulation_multicontrol_spec.json \
  -o outputs/openmdw_multicontrol --model=multicontrol/openmdw-1.1
```

For multi-GPU inference on a single control or to run multiple control variants, use [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):
```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i assets/coagulation_example/depth/coagulation_depth_spec.json -o outputs/depth
```

We provide example parameter files for each individual control variant along with a multi-control variant:

| Variant | Parameter File  |
| --- | --- |
| Depth | `assets/coagulation_example/depth/coagulation_depth_spec.json` |
| Edge | `assets/coagulation_example/edge/coagulation_edge_spec.json` |
| Segmentation | `assets/coagulation_example/seg/coagulation_seg_spec.json` |
| Blur | `assets/coagulation_example/vis/coagulation_vis_spec.json` |
| Multi-control | `assets/coagulation_example/multicontrol/coagulation_multicontrol_spec.json` |

For an explanation of all the available parameters run:
```bash
python examples/inference.py --help

python examples/inference.py control:edge --help # for information specific to edge control
```

Parameters can be specified as json:

```jsonc
{
    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/coagulation_example/coagulation_prompt.json",

    // Directory to save the generated video
    "output_dir": "outputs/coagulation_multicontrol",

    // Path to the input video
    "video_path": "assets/coagulation_example/coagulation_input.mp4",

    // Inference settings:
    "guidance": 3,

    // Depth control settings
    "depth": {
        // Path to the control video
        // If a control is not provided, it will be computed on the fly.
        "control_path": "assets/coagulation_example/depth/coagulation_depth.mp4",

        // Control weight for the depth control
        "control_weight": 1.0
    },

    // Edge control settings
    "edge": {
        // Control video computed on the fly
        // Default control weight of 1.0 for edge control
    },

    // Seg control settings
    "seg": {
        // Path to the control video
        "control_path": "assets/coagulation_example/seg/coagulation_seg.mp4",

        // Control weight for the seg control
        "control_weight": 1.0
    },

    // Blur control settings
    "vis":{
        // Control video computed on the fly
        "control_weight": 1.0
    }
}
```

If you would like the control inputs to only be used for some regions, you can define binary spatiotemporal masks with the corresponding control input modality in mp4 format. White pixels means the control will be used in that region, whereas black pixels will not. Example below:


```jsonc
{
    "depth": {
        "control_path": "assets/coagulation_seg/depth/coagulation_depth.mp4",
        "mask_path": "/path/to/depth/mask.mp4",
        "control_weight": 0.5
    }
}
```

### Example Input

<table>
  <tr>
    <th>Depth Control</th>
    <th>Edge Control</th>
    <th>Seg Control</th>
    <th>Vis Control</th>
  </tr>
  <tr>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/35817783-5a61-4302-a38d-fcd7c32944d9" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/253afe1b-55ad-417c-9a4f-301dc1fb852c" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/0e69bf77-cb8f-48ae-9022-3200fedf7588" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/0850a4b7-e996-4dff-840c-9249d70a1292" width="100%" controls></video>
    </td>
  </tr>
</table>

### Example Output
<table>
  <tr>
    <th>Depth Control Output</th>
    <th>Edge Control Output</th>
    <th>Seg Control Output</th>
    <th>Vis Control Output</th>
  </tr>
  <tr>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/e817b143-3af3-4219-8a3f-479b99f6c484" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/2219e1d3-f428-4f98-bc89-5af1d66af962" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/ed62210a-f20a-4a6a-8206-d526a2c6bf72" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/61d056b6-c92a-478a-b522-a5f00c5eb31c" width="100%" controls></video>
    </td>
  </tr>
</table>

<table>
  <tr>
    <th>Output Multi-control Video</th>
  </tr>
  <tr>
    <td valign="middle" width="50%">
      <video src="https://github.com/user-attachments/assets/a712073a-40d3-4ed7-884c-416695ec4f55" width="50%" controls></video>
    </td>
  </tr>
</table>


### Partial Denoising

Partial denoising starts generation from a noised version of the input video latent instead of starting from pure random noise. Use the optional `sigma_max` field to control how much noise is added to the encoded input before denoising.

```jsonc
{
    "name": "coagulation_depth_partial_denoising_70",
    "prompt_path": "coagulation_example/coagulation_prompt.txt",
    "video_path": "coagulation_example/coagulation_input.mp4",
    "guidance": 3,

    // Current JSON/JSONL schema expects sigma_max as a string.
    "sigma_max": "70",

    "depth": {
        "control_path": "coagulation_example/depth/coagulation_depth.mp4",
        "control_weight": 1.0
    }
}
```

For [JSONL batch testing](../assets/depth_partial_denoising.jsonl), run the same sample with several `sigma_max` values and compare the outputs.

In this rectified-flow pipeline, the user-facing `sigma_max` is converted to a normalized flow sigma by `sigma_max / (1 + sigma_max)`. 

Values are accepted in the documented range `"0"` to `"200"`, but high values such as larger than `"100"` are all close to full-noise in the rectified-flow schedule. Lower values are usually more useful when testing input preservation.

#### Example Output with Different `sigma_max`

<table>
  <tr>
    <th>Original Video</th>
    <th>Depth Control Output (sigma_max=10)</th>
    <th>Depth Control Output (sigma_max=70)</th>
    <th>Depth Control Output (sigma_max=200)</th>
  </tr>
  <tr>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/ec1c13c4-ca34-43b0-9ec1-3b795ed1981a" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/b6c1dd1c-0489-4e52-86da-a5a90568ad6e" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/3c6fec63-4541-4a10-acae-2f28358c59e7" width="100%" controls></video>
    </td>
    <td valign="middle" width="25%">
      <video src="https://github.com/user-attachments/assets/38b42a45-53c3-4401-b656-c12fc9f6eccc" width="100%" controls></video>
    </td>
  </tr>
</table>
