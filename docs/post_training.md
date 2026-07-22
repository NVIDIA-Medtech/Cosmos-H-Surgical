# Post-Training Preview

Post-training is a developer preview in v0.3.0. The supported release
capabilities are I2V prediction and transfer inference; the included recipe is
provided to make the surgical configuration and dataset contract auditable.

## Included Recipe

`examples/post_training/cosmos_h_surgical_vision_lora_480p.toml` trains a
Cosmos3-Nano LoRA for surgical T2V and I2V using the public Cosmos Framework
JSONL loader.

Set the required paths:

```bash
export COSMOS_H_SURGICAL_DATASET=/absolute/path/to/surgical_train.jsonl
export BASE_CHECKPOINT_PATH=/absolute/path/to/cosmos3_base_checkpoint
export WAN_VAE_PATH=/absolute/path/to/Wan2.2_VAE.pth
```

Install one CUDA group and launch eight processes:

```bash
uv sync --group cu130
uv run --no-sync torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical.training \
  --sft-toml examples/post_training/cosmos_h_surgical_vision_lora_480p.toml
```

Use `--group cu128` instead when the environment targets CUDA 12.8.

## Dataset Contract

The JSONL follows the public Cosmos Framework SFT schema. Each record provides:

- `uuid`
- `vision_path`
- Video dimensions and timing
- One or more `t2w_windows`

Structured surgical captions may be stored as `caption_json` inside each
window. Dataset paths must be portable and must not rely on internal object
stores or unpublished adapters.

## Checkpoints and Outputs

`BASE_CHECKPOINT_PATH` must point to a compatible Cosmos 3 base checkpoint.
`WAN_VAE_PATH` must point to the VAE expected by the TOML configuration.
Training output and checkpoint policies are controlled by the recipe's
`[checkpoint]` and `[trainer]` sections.

Before publishing a trained result, record:

- Base checkpoint revision and digest
- Dataset version and license
- Full TOML configuration
- Cosmos-H-Surgical and framework revisions
- GPU/CUDA environment
- Final checkpoint inventory and digest

## Current Limitations

The mixed transfer/action recipe used during internal development is not part
of the public v0.3.0 interface. It requires public dataset adapters,
control-sidecar documentation, and clean-data validation before release.

For the generic trainer and structured TOML schema, consult the pinned upstream
guides:

- [Post-training](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/training.md)
- [SFT configuration](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/sft_config.md)
