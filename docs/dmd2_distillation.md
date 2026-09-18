# Four-Step DMD2 Distillation

Cosmos-H-Surgical v0.3.1 includes a DMD2 recipe for distilling surgical
video-to-video transfer into a four-denoiser-call student. The public recipe is
the recommended v0.3.1 starting point for new distillation runs; it is not an
exact reconstruction of the internal run used to produce the released
checkpoint.

## Single-H100 Inference Latency

The released DMD2 student and the original 50-step Cosmos-H-Surgical transfer
checkpoint were benchmarked on one NVIDIA H100 80GB using the same edge input,
seed 0, BF16, 832 x 480 output, 93 frames, and 16 FPS, with compilation
disabled. One warmup was followed by five measured repetitions. Values are
mean +/- sample standard deviation in seconds.

| Stage | DMD2 four-step | Base 50-step | Base / DMD2 |
| --- | ---: | ---: | ---: |
| VAE encode CUDA, two calls total | 1.107 +/- 0.000 | 1.108 +/- 0.000 | 1.00x |
| Denoising CUDA | 4.087 +/- 0.007 | 107.163 +/- 0.389 | 26.22x |
| VAE decode CUDA, one call | 1.993 +/- 0.002 | 1.994 +/- 0.002 | 1.00x |
| Generation wall time | 8.780 +/- 0.048 | 112.704 +/- 0.415 | 12.84x |

The DMD2 student made four denoiser calls and did not execute an additional
classifier-free guidance branch. The base model made 100 denoiser calls: two
guidance evaluations for each of 50 sampling steps. These measurements are a
reference for this configuration; latency varies with hardware, software, and
input or output settings.

The default recipe samples four transfer controls with equal probability:

| Control | Training source |
| --- | --- |
| Edge | Computed from the RGB target during loading. |
| Blur | Loaded from an aligned sidecar when present, otherwise computed during loading. |
| Depth | Loaded from an aligned depth sidecar. |
| Segmentation | Loaded from an aligned segmentation sidecar. |

The student uses the fixed SDE schedule
`[1.0, 0.9375, 0.8333333333333334, 0.625]`; sampling appends the terminal zero
step. The released inference checkpoint uses text guidance `1.0` and control
guidance `1.0`, so inference does not execute an additional classifier-free
guidance branch.

## Requirements

Complete the standard [setup](setup.md) first. The default training topology
uses eight GPU ranks per node and BF16 precision. All nodes in a multi-node run
must see the same repository, datasets, teacher DCP, VAE, and output directory.

The following tools must be on `PATH`:

- `ffmpeg` and `ffprobe`
- the project `.venv` created by `uv sync`

## Prepare the Teacher and VAE

The released checkpoint is distributed as Hugging Face safetensors. Cosmos
Framework training consumes PyTorch Distributed Checkpoint (DCP), so prepare a
private DCP copy once for training. The DCP is not a public inference artifact
and should not be uploaded with a release.

```bash
export HF_HOME=/path/to/huggingface-cache
export COSMOS_H_SURGICAL_HF_REPOSITORY=nvidia/Cosmos-H-Surgical
export COSMOS_H_SURGICAL_HF_REVISION=v0.3.1

export RELEASE_CHECKPOINT_PATH="$(
  hf download \
    "$COSMOS_H_SURGICAL_HF_REPOSITORY" \
    --revision "$COSMOS_H_SURGICAL_HF_REVISION"
)"

cosmos-h-surgical validate-distillation-teacher \
  --checkpoint-path "$RELEASE_CHECKPOINT_PATH"

export COSMOS_H_SURGICAL_DISTILL_TEACHER_DCP="$PWD/checkpoints/Cosmos-H-Surgical-teacher.dcp"
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$RELEASE_CHECKPOINT_PATH" \
  -o "$COSMOS_H_SURGICAL_DISTILL_TEACHER_DCP"

export WAN_VAE_PATH="$(
  hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth
)"
```

The teacher preflight checks the released LoRA inventory and rejects missing or
empty safetensors shards. Training performs an independent DCP compatibility
check before copying the teacher into the student and fake-score networks.

## Prepare Transfer Data

Use the portable dataset contract from [Surgical post-training](post_training.md).
Every manifest contains RGB target videos relative to a dataset root. Caption,
blur, depth, and segmentation sidecars share each target's stem.

Configure one dataset:

```bash
export COSMOS_H_SURGICAL_TRANSFER_DATASET_DIRS=/data/surgical_dataset
export COSMOS_H_SURGICAL_TRANSFER_JSON_PATHS=/data/surgical_dataset/manifests/train.json
export COSMOS_H_SURGICAL_TRANSFER_ENLARGED_FACTORS=1.0
```

For multiple datasets, provide comma-separated values in the same order. The
three variables must contain the same number of entries.

Validate every source before allocating training GPUs:

```bash
cosmos-h-surgical validate-training-data \
  --mode transfer \
  --dataset-dir /data/surgical_dataset \
  --manifest /data/surgical_dataset/manifests/train.json \
  --control-modalities edge,blur,depth,seg
```

The repository toy dataset can exercise configuration and data loading:

```bash
export COSMOS_H_SURGICAL_TRANSFER_DATASET_DIRS="$PWD/datasets/cosmos-h-surgical-assets"
export COSMOS_H_SURGICAL_TRANSFER_JSON_PATHS="$PWD/datasets/cosmos-h-surgical-assets/manifests/train.json"
export COSMOS_H_SURGICAL_TRANSFER_ENLARGED_FACTORS=1.0
```

## Train on One Node

The DMD2 recipe follows the same direct `torchrun` pattern as the Predict and
Transfer post-training recipes. Set a stable run name and shared output root,
then start eight ranks:

```bash
export DMD2_RUN_NAME=transfer_dmd2_example
export IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/dmd2_train"
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_transfer_dmd2_lora_480p_4step.toml \
  job.name="$DMD2_RUN_NAME" \
  job.wandb_mode=disabled \
  trainer.max_iter=8000 \
  trainer.seed=42 \
  checkpoint.save_iter=50 \
  2>&1 | tee "$IMAGINAIRE_OUTPUT_ROOT/train_dmd2.log"
```

Change the final arguments to override the run name, logging mode, iteration
count, seed, checkpoint interval, or other Hydra settings. For example, append
`trainer.logging_iter=5` for more frequent logging.

Before allocating GPUs, run the data validator shown above and confirm that
`$COSMOS_H_SURGICAL_DISTILL_TEACHER_DCP/model/.metadata` and `$WAN_VAE_PATH`
exist and are nonempty.

## Train on Multiple Nodes

Have the scheduler run the following command once per node with the same
rendezvous values and run name:

```bash
export NNODES=4
export NODE_RANK=0  # Set to 0, 1, 2, or 3 on the corresponding node.
export MASTER_ADDR=training-node-0
export MASTER_PORT=29500
export DMD2_RUN_NAME=transfer_dmd2_example
export IMAGINAIRE_OUTPUT_ROOT=/shared/outputs/dmd2_train

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node=8 \
  --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id="$DMD2_RUN_NAME" \
  -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_transfer_dmd2_lora_480p_4step.toml \
  job.name="$DMD2_RUN_NAME" \
  job.wandb_mode=disabled \
  trainer.max_iter=8000 \
  trainer.seed=42 \
  checkpoint.save_iter=50 \
  2>&1 | tee "$IMAGINAIRE_OUTPUT_ROOT/train_dmd2_node_${NODE_RANK}.log"
```

The scheduler is responsible for assigning `NODE_RANK`, selecting the
rendezvous host, and making the repository, datasets, teacher DCP, VAE, and
output root visible at identical paths on every node.

## Resume Training

Resume only from a complete DMD2 checkpoint produced with the same world size:

```bash
export DMD2_RUN_NAME=transfer_dmd2_example
export DMD2_RESUME_CHECKPOINT="$IMAGINAIRE_OUTPUT_ROOT/cosmos_h_surgical/transfer_dmd2_lora_480p_4step/transfer_dmd2_example/checkpoints/iter_000001000"

torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_transfer_dmd2_lora_480p_4step.toml \
  job.name="$DMD2_RUN_NAME" \
  checkpoint.load_path="$DMD2_RESUME_CHECKPOINT" \
  checkpoint.load_training_state=true \
  checkpoint.strict_resume=true \
  2>&1 | tee "$IMAGINAIRE_OUTPUT_ROOT/train_dmd2_resume.log"
```

The checkpoint must contain the model, student and fake-score optimizer,
scheduler, and trainer metadata and shards. Strict resume rejects partial or
incompatible state rather than silently warm-starting.

## Recipe Details

The recipe uses:

- A rank-16 LoRA with alpha 32 on generation Q/K/V/O projections.
- Backward fixed-step SDE simulation.
- One gradient-carrying rollout step.
- Training-only teacher guidance `3.0`.
- Independent student and fake-score LoRA optimizers at `5e-5`.
- An 8,000-iteration schedule with 100 warmup steps.
- Per-iteration JSONL metrics and optional W&B reporting.

Only LoRA parameters are optimized. Teacher weights are frozen. The saved DCP
contains full training state so a stopped job can resume exactly.

## Export a Student for Inference

Select a complete iteration and export only the student to a portable
safetensors directory:

```bash
export RUN_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos_h_surgical/transfer_dmd2_lora_480p_4step/transfer_dmd2_example"
export CHECKPOINT_PATH="$RUN_DIR/checkpoints/iter_000008000"
export EXPORT_PATH="$RUN_DIR/model"

python -m cosmos_h_surgical.export_dmd2 \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --config-file "$RUN_DIR/config.yaml" \
  -o "$EXPORT_PATH"
```

The wrapper invokes the pinned framework's student-only, no-ViT exporter,
replaces run-local tokenizer and model-config paths with public aliases, and
runs a single-GPU inference smoke test. The export contains the standalone
student model, configuration, safetensors index and shards, and portable
metadata. It excludes training-only networks, optimizers, schedulers, trainer
state, and source paths.

## Run the Released Student

The v0.3.1 release registers the DMD2 student as
`Cosmos-H-Surgical-Transfer-DMD2-4Step`:

```bash
python - <<'PY'
import json
from pathlib import Path

source = Path("inputs/transfer/specs/edge_coagulation.json")
sample = json.loads(source.read_text())
sample["guidance"] = 1.0
sample["control_guidance"] = 1.0
Path("/tmp/dmd2_edge_coagulation.json").write_text(json.dumps(sample, indent=2) + "\n")
PY

torchrun --nproc-per-node=1 \
  -m cosmos_h_surgical infer \
  --checkpoint-path Cosmos-H-Surgical-Transfer-DMD2-4Step \
  --parallelism-preset=latency \
  --dp-shard-size=1 \
  --no-use-torch-compile \
  -i /tmp/dmd2_edge_coagulation.json \
  -o outputs/dmd2-transfer \
  --no-guardrails \
  --seed=0
```

To validate a local export before publication, replace the registered model
name with `"$EXPORT_PATH"`.
