# Post-Training Recipes

`cosmos_h_surgical_vision_lora_480p.toml` is the first public Cosmos 3 recipe.
It trains a Cosmos3-Nano LoRA for surgical T2V and I2V using the public Cosmos
Framework JSONL loader. Set these variables before launching:

```bash
export COSMOS_H_SURGICAL_DATASET=/absolute/path/to/surgical_train.jsonl
export BASE_CHECKPOINT_PATH=/absolute/path/to/cosmos3_base_checkpoint
export WAN_VAE_PATH=/absolute/path/to/Wan2.2_VAE.pth

uv sync --frozen --extra train
uv run torchrun --nproc_per_node=8 -m cosmos_h_surgical.training \
  --sft-toml examples/post_training/cosmos_h_surgical_vision_lora_480p.toml
```

The JSONL must follow the public Cosmos Framework SFT schema. In particular,
each record supplies `uuid`, `vision_path`, video dimensions and timing, plus
one or more `t2w_windows`; structured surgical captions may use
`caption_json` inside each window.

The current release candidate's mixed transfer/action recipe is not copied
from the internal branch. It will be added after its public dataset adapters,
control-sidecar contract, and clean-data smoke tests are ready.
