# Cosmos-H-Surgical-Transfer: World Generation with Guided Generation
This guide provides instructions on running inference with guided generation. The Guided generation enables domain
randomization by transferring simulation videos to realistic-looking footage while maintaining structural consistency
through various control inputs, without any additional model training. Instead of allowing the model to
freely reinterpret the entire scene, we encode simulation frames into the model’s latent space and apply spatial
constraints during the denoising process. This selectively anchors important regions—such as surgical tools—while leaving the rest of the scene unconstrained. As a result, the model can enhance
global realism (lighting, textures, background complexity) while preserving the geometric structure and
identity of critical foreground elements.

### Pre-requisites
Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

## Guided Generation Inference with Pre-trained Cosmos-H-Surgical-Transfer Models

Individual control variants can be run on a single GPU:
```bash
python examples/inference.py -i transfer/assets/needlePuncture_guided_example/seg/needlePuncture_guided_seg_spec.json -o outputs/seg_guided_generation
```

For multi-GPU inference on a single control or to run multiple control variants, use [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):
```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i transfer/assets/needlePuncture_guided_example/seg/needlePuncture_guided_seg_spec.json -o outputs/seg_guided_generation
```

We provide a example parameter files for individual control:

| Variant | Parameter File  |
| --- | --- |
| Segmentation | `assets/humanoid_example/seg/humanoid_seg_guided_spec.json` |

For an explanation of all the available parameters run:
```bash
python examples/inference.py --help

python examples/inference.py control:seg --help # for information specific to seg control
```

Parameters can be specified as json:

```jsonc
{
    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/needlePuncture_guided_example/needlePuncture_guided_prompt.txt",

    // Directory to save the generated video
    "output_dir": "outputs/needlePuncture_guided_generation",

    // Path to the input video
    "video_path": "assets/needlePuncture_guided_example/needlePuncture_guided_input.mp4",

    // Path to the binary mask video indicating the foreground elements for guided generation
    "guided_generation_mask": "assets/needlePuncture_guided_example/needlePuncture_guided_input.mp4",

    // Number of steps for guided generation. Using more guidance steps provides stronger guidance of foreground
    // elements in the generated videos. By default, 25 steps are used for guided generation.
    "guided_generation_step_threshold": 25,

    // Inference settings:
    "guidance": 3,

    // Seg control settings
    "seg": {
        // Path to the control video
        "control_path": "assets/needlePuncture_guided_example/seg/needlePuncture_guided_seg.mp4",

        // Control weight for the seg control
        "control_weight": 0.5
    }
}
```

### Example Input

<table>
  <tr>
    <th>Input Video</th>
    <th>Seg Control</th>
    <th>Guided Generation Mask</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/fbc14bfc-d465-42d0-960a-d02664ec097d" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/c66e7533-0d7f-47d9-a9fe-3fa34f58b1c8" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/658c60bd-6307-45ca-a47a-4d8096608d9e" width="100%" controls></video>
    </td>
  </tr>
</table>

### Example Output
<table>
  <tr>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/c707ec7e-4e32-48cd-97fa-e970636a78d1" width="60%" controls></video>
    </td>
  </tr>
</table>

