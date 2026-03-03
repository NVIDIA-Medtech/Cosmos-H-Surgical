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
    <th>Output Multi-control Video</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/a712073a-40d3-4ed7-884c-416695ec4f55" width="60%" controls></video>
    </td>
  </tr>
</table>
<table>
  <tr>
    <th>Output Depth-control Video</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/e817b143-3af3-4219-8a3f-479b99f6c484" width="60%" controls></video>
    </td>
  </tr>
</table>
<table>
  <tr>
    <th>Output Edge-control Video</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/2219e1d3-f428-4f98-bc89-5af1d66af962" width="60%" controls></video>
    </td>
  </tr>
</table>
<table>
  <tr>
    <th>Output Seg-control Video</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/ed62210a-f20a-4a6a-8206-d526a2c6bf72" width="60%" controls></video>
    </td>
  </tr>
</table>
<table>
  <tr>
    <th>Output Vis-control Video</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/61d056b6-c92a-478a-b522-a5f00c5eb31c" width="60%" controls></video>
    </td>
  </tr>
</table>