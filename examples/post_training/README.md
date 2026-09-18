# Post-Training Recipes

The public Cosmos 3 release contains two independent Cosmos3-Nano 480P LoRA
recipes:

- `cosmos_h_surgical_predict_lora_480p.toml` for surgical T2V, first-frame I2V,
  and short-continuation I2V.
- `cosmos_h_surgical_transfer_lora_480p.toml` for edge, blur, depth, and
  segmentation transfer.

Both recipes use development-format manifests containing relative target-video
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
    --revision v0.3.0
)"
export WAN_VAE_PATH="$(
  hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth
)"
export BASE_CHECKPOINT_PATH="$PWD/checkpoints/Cosmos-H-Surgical-v0.3.0"

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

See [Surgical post-training](../../docs/post_training.md) for the exact
manifest, caption, control-sidecar, preparation, validation, and DCP-to-Hugging
Face export contracts.

## Four-Step Transfer Distillation

`cosmos_h_surgical_transfer_dmd2_lora_480p_4step.toml` distills the released
Cosmos-H-Surgical transfer behavior into a four-step SDE student. The release
is a Nano base plus a trained rank-16 LoRA, so the teacher, student, and
fake-score networks all include that adapter. Only the student and fake-score
LoRA copies are optimized; the teacher is frozen.

Preflight the downloaded release before its one-time DCP conversion:

```bash
export RELEASE_CHECKPOINT_PATH=/path/to/Cosmos-H-Surgical/snapshot

cosmos-h-surgical validate-distillation-teacher \
  --checkpoint-path "$RELEASE_CHECKPOINT_PATH"

export COSMOS_H_SURGICAL_DISTILL_TEACHER_DCP="$PWD/checkpoints/Cosmos-H-Surgical-release.dcp"
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$RELEASE_CHECKPOINT_PATH" \
  -o "$COSMOS_H_SURGICAL_DISTILL_TEACHER_DCP"
```

The preflight requires the released 288-tensor LoRA inventory and all nonempty
safetensors shards. At training startup, the DMD2 loader independently requires
every transfer-relevant DCP tensor and permits only the five release action-head
tensors to be unused by this transfer-only recipe.

Launch with the standard transfer dataset variables and VAE path:

```bash
torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_transfer_dmd2_lora_480p_4step.toml
```

The repository also includes a guarded runner for an existing eight-GPU
interactive container. It defaults to the `surgical_syn_mixed` dataset preset;
choose a unique run name and launch from the repository root:

```bash
export DMD2_RUN_NAME=cosmos_h_transfer_dmd2_dev

./run_cosmos_h_surgical_transfer_dmd2.sh
```

The preset reproduces the 15 dataset roots, manifests, and enlargement factors
from the Cosmos Framework `cosmos_h_surg_mixed_capability_syn_480p` recipe. On
the host, those datasets live under `/home/pengfeig/healthcareeng_monai/datasets`.
The interactive container must mount `/home/pengfeig/healthcareeng_monai` at
`/healthcareeng_monai`; all preset paths use the container-visible location.

Set `DMD2_USE_TOY_DATA=1 DMD2_DRY_RUN=1` for a no-training environment and
data preflight. Resume requires a complete native eight-GPU DMD2 checkpoint:

```bash
export DMD2_RUN_NAME=cosmos_h_transfer_dmd2_dev
export DMD2_RESUME_CHECKPOINT=/workspace/code/Cosmos-H-Surgical/outputs/dmd2_train/cosmos_h_surgical/transfer_dmd2_lora_480p_4step/cosmos_h_transfer_dmd2_dev/checkpoints/iter_000000050

./run_cosmos_h_surgical_transfer_dmd2.sh
```

For a production allocation, export the same variables on the login node and
submit the Slurm wrapper. Dataset and resume paths must be visible inside the
container under `/workspace` or `/healthcareeng_monai`. The Slurm wrapper adds
the required dataset mount and defaults to `DMD2_DATASET_PRESET=surgical_syn_mixed`:

```bash
sbatch --export=ALL train_cosmos_h_surgical_transfer_dmd2.slurm
```

`DMD2_MAX_ITER`, `DMD2_SAVE_ITER`, `DMD2_SEED`, `DMD2_WANDB_MODE`,
`IMAGINAIRE_OUTPUT_ROOT`, and additional Hydra overrides passed as script
arguments are supported by both launch paths. The Slurm launcher defaults to
four eight-GPU nodes (32 ranks total) on `batch` for four hours; `sbatch`
command-line options can override the job name, partition, or wall time. It
starts one containerized `torchrun` agent per node and derives the rendezvous
host, port, and node rank from the Slurm allocation.

The student schedule is `[1.0, 0.9375, 0.8333333333333334, 0.625]`; the sampler
adds the final zero step. The recipe uses backward simulation, one gradient
carrying rollout step, teacher guidance 3.0, and independent `5e-5` LoRA-only
optimizers for the student and fake-score networks. The runner defaults to
`DMD2_WANDB_MODE=online`; set it to `offline` or `disabled` explicitly when
needed. `trainer.logging_iter` controls both W&B updates and rank-zero iteration
speed lines. With the default value of one, stdout reports every iteration after
the first timing sample, while `dmd2_metrics.jsonl` records every completed step.
