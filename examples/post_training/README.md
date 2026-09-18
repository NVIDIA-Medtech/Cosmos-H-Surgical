# Post-Training Recipes

The public Cosmos 3 release contains independent Cosmos3-Nano 480P LoRA
recipes:

- `cosmos_h_surgical_predict_lora_480p.toml` for surgical T2V, first-frame I2V,
  and short-continuation I2V.
- `cosmos_h_surgical_transfer_lora_480p.toml` for edge, blur, depth, and
  segmentation transfer.
- `cosmos_h_surgical_transfer_dmd2_lora_480p_4step.toml` for four-step DMD2
  transfer distillation.

All recipes use development-format manifests containing relative target-video
paths plus adjacent caption and control sidecars. Dataset roots, manifests, and
enlargement factors are supplied through environment variables; no storage
paths are embedded in the repository.

Predict example:

```bash
uv sync --group cu130
source .venv/bin/activate

export COSMOS_H_SURGICAL_PREDICT_DATASET_DIRS=/data/surgical_dataset
export COSMOS_H_SURGICAL_PREDICT_JSON_PATHS=/data/surgical_dataset/manifests/train.json
export COSMOS_H_SURGICAL_PREDICT_ENLARGED_FACTORS=1.0
export HF_HOME=/path/to/huggingface-cache
export RELEASE_CHECKPOINT_PATH="$(
  hf download nvidia/Cosmos-H-Surgical \
    --revision v0.3.1
)"
export WAN_VAE_PATH="$(
  hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth
)"
export BASE_CHECKPOINT_PATH="$PWD/checkpoints/Cosmos-H-Surgical-v0.3.1"

python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$RELEASE_CHECKPOINT_PATH" \
  -o "$BASE_CHECKPOINT_PATH"

torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_predict_lora_480p.toml
```

The one-time conversion preserves the released surgical LoRA in a DCP that the
training recipe can load directly. Reuse `BASE_CHECKPOINT_PATH` for later runs;
do not repeat conversion on every launch.

Transfer uses the corresponding `COSMOS_H_SURGICAL_TRANSFER_*` variables and
`cosmos_h_surgical_transfer_lora_480p.toml`.

See [Surgical post-training](../../docs/post_training.md) for the manifest,
caption, control-sidecar, preparation, validation, and export contracts. See
[Four-step DMD2 distillation](../../docs/dmd2_distillation.md) for the complete
distillation tutorial, resume flow, and student-only export.
