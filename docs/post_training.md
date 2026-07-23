# Surgical Post-Training

Cosmos-H-Surgical provides separate Cosmos3-Nano 480P LoRA recipes for
Predict and Transfer. They use the same manifest and sidecar organization as
the development training pipeline, with all storage locations supplied by the
user.

These capability-specific recipes do not exactly reproduce the release
checkpoint's joint Predict, Transfer, and Action training run. Action training
is outside the v0.3.0 public interface.

The pinned Cosmos Framework video loader requires `ffmpeg` and `ffprobe` on
`PATH` during training. The dataset validation command uses the PyAV dependency
installed by the project environment and does not require a separate probe
binary.

## Dataset Layout

Each manifest entry identifies an RGB target video relative to a dataset root.
The caption and transfer controls are adjacent sidecars with the same stem:

```text
surgical_dataset/
|-- manifests/
|   `-- train.json
`-- videos/
    |-- clip_000001.mp4
    |-- clip_000001.json
    |-- clip_000001.blur.mp4
    |-- clip_000001.depth.mp4
    `-- clip_000001.seg.mp4
```

Use portable relative paths in the manifest:

```json
{
  "training": [
    {"video": "videos/clip_000001.mp4"},
    {"video": "videos/clip_000002.mp4"}
  ]
}
```

A top-level list of strings or objects is also accepted for compatibility.
The canonical form above is recommended for new datasets.

The release recipes select the structured `caption_json` field from each
adjacent caption file:

```json
{
  "caption_json": {
    "description": "A structured description of the surgical clip."
  }
}
```

The value may contain the complete structured prompt produced by the prompt
upsampler. See [Prompt upsampling](prompt_upsampling.md) for video-level prompt
generation.

## Transfer Controls

The Transfer recipe samples edge, blur, depth, and segmentation controls with
equal weight:

| Control | Training source |
| --- | --- |
| Edge | Computed from the RGB target at load time. |
| Blur | Uses `.blur.mp4` when present; otherwise computed at load time. |
| Depth | Requires an aligned `.depth.mp4`. |
| Segmentation | Requires an aligned `.seg.mp4`. |

Depth and segmentation controls must cover the same frames as the RGB target.
For clean-data validation, use the same dimensions, frame count, and FPS for
the target and all precomputed controls.

## Included Toy Dataset

The repository includes ten training-ready synthetic examples under
`datasets/cosmos-h-surgical-assets`. The RGB targets are synthetic videos
selected from the corresponding action examples in the synthetic training
corpus and published under action-only filenames. Each target has an adjacent
structured caption plus aligned blur, depth, and segmentation sidecars. Edge
control is computed from the RGB target at load time.

The same portable manifest works with both release recipes:

```bash
export TOY_DATASET_DIR="$PWD/datasets/cosmos-h-surgical-assets"
export TOY_DATASET_MANIFEST="$TOY_DATASET_DIR/manifests/train.json"

cosmos-h-surgical validate-training-data \
  --mode predict \
  --dataset-dir "$TOY_DATASET_DIR" \
  --manifest "$TOY_DATASET_MANIFEST"

cosmos-h-surgical validate-training-data \
  --mode transfer \
  --dataset-dir "$TOY_DATASET_DIR" \
  --manifest "$TOY_DATASET_MANIFEST" \
  --control-modalities edge,blur,depth,seg
```

Point the `COSMOS_H_SURGICAL_PREDICT_*` or
`COSMOS_H_SURGICAL_TRANSFER_*` environment variables below at these two paths
for a small end-to-end recipe smoke test. The dataset README documents its
files and usage.

## Prepare a Manifest

The preparation command accepts recursive patterns relative to the dataset
root. Transfer sidecars are excluded automatically:

```bash
cosmos-h-surgical prepare-training-data \
  --dataset-dir /data/surgical_dataset \
  --video-pattern "videos/**/*.mp4" \
  --output /data/surgical_dataset/manifests/train.json
```

The command requires an adjacent JSON caption for every selected target. Use
`--allow-missing-captions` only while inspecting an incomplete dataset.

Validate Predict data:

```bash
cosmos-h-surgical validate-training-data \
  --mode predict \
  --dataset-dir /data/surgical_dataset \
  --manifest /data/surgical_dataset/manifests/train.json
```

Validate Transfer data and all four controls:

```bash
cosmos-h-surgical validate-training-data \
  --mode transfer \
  --dataset-dir /data/surgical_dataset \
  --manifest /data/surgical_dataset/manifests/train.json \
  --control-modalities edge,blur,depth,seg
```

The validator checks manifest structure, portable paths, captions, minimum
frame counts, required controls, and target/control media alignment.

## Dataset Sources

The loader accepts one or more dataset roots, manifest paths, and enlargement
factors. Comma-separated values are paired by position and must have equal
length:

```bash
export COSMOS_H_SURGICAL_PREDICT_DATASET_DIRS="/data/set_a,/data/set_b"
export COSMOS_H_SURGICAL_PREDICT_JSON_PATHS="/data/set_a/manifests/train.json,/data/set_b/manifests/train.json"
export COSMOS_H_SURGICAL_PREDICT_ENLARGED_FACTORS=1.0,0.5
```

An enlargement factor repeats or subsamples a source before the combined
dataset is shuffled. Use `1.0` unless intentional source reweighting is needed.
Transfer uses the corresponding `COSMOS_H_SURGICAL_TRANSFER_*` variables.

## Prepare the Training Checkpoint and VAE

Cosmos Framework trains VFM recipes from a PyTorch Distributed Checkpoint
(DCP), while released inference checkpoints use Hugging Face safetensors. New
Predict and Transfer runs therefore use this one-time preparation flow:

1. Download the complete released Cosmos-H-Surgical snapshot.
2. Convert the snapshot to DCP with the pinned framework converter.
3. Point `BASE_CHECKPOINT_PATH` at that DCP for training.

The complete snapshot is required because the converter reads the model
configuration as well as the safetensor shards. Do not restrict this download
to `*.safetensors` files.

```bash
uv sync --group cu130
source .venv/bin/activate

export HF_HOME=/path/to/huggingface-cache
export COSMOS_H_SURGICAL_HF_REPOSITORY=nvidia/Cosmos-H-Surgical
export COSMOS_H_SURGICAL_HF_REVISION=v0.3.0

export RELEASE_CHECKPOINT_PATH="$(
  hf download \
    "$COSMOS_H_SURGICAL_HF_REPOSITORY" \
    --revision "$COSMOS_H_SURGICAL_HF_REVISION"
)"

export WAN_VAE_PATH="$(
  hf download \
    Wan-AI/Wan2.2-TI2V-5B \
    Wan2.2_VAE.pth
)"
```

Use `uv sync --group cu128` instead when the environment targets CUDA 12.8.

Both downloads are cached under `HF_HOME`; use the paths printed by the CLI
instead of constructing snapshot paths manually. Convert the downloaded model
once and keep the DCP on storage visible to every training rank:

```bash
export BASE_CHECKPOINT_PATH="$PWD/checkpoints/Cosmos-H-Surgical"

python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$RELEASE_CHECKPOINT_PATH" \
  -o "$BASE_CHECKPOINT_PATH"
```

The converted directory contains `model/.metadata`, model `*.distcp` shards,
and model configuration metadata. Both public TOMLs read
`[checkpoint].load_path` from `BASE_CHECKPOINT_PATH`, so distributed training
loads the prepared DCP directly. The DCP preserves the released surgical LoRA
weights; the recipes continue optimizing those adapters instead of creating a
base-only initialization.

To test the private release candidate before publication, set
`COSMOS_H_SURGICAL_HF_REPOSITORY=pengfeig/Cosmos-H-Surgical-staging` and the
approved release-candidate revision before running the same commands.

## Predict Training

Set the Predict dataset variables:

```bash
export COSMOS_H_SURGICAL_PREDICT_DATASET_DIRS=/data/surgical_dataset
export COSMOS_H_SURGICAL_PREDICT_JSON_PATHS=/data/surgical_dataset/manifests/train.json
export COSMOS_H_SURGICAL_PREDICT_ENLARGED_FACTORS=1.0
export IMAGINAIRE_OUTPUT_ROOT="$PWD/outputs/train"
mkdir -p $IMAGINAIRE_OUTPUT_ROOT
```

Launch eight processes:

```bash
torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_predict_lora_480p.toml \
  2>&1 | tee "$IMAGINAIRE_OUTPUT_ROOT/train_predict.log"
```

The Predict recipe samples 20% text-only, 70% first-frame I2V, and 10%
short-continuation I2V conditioning.

## Transfer Training

Set the independent Transfer source variables:

```bash
export COSMOS_H_SURGICAL_TRANSFER_DATASET_DIRS=/data/surgical_dataset
export COSMOS_H_SURGICAL_TRANSFER_JSON_PATHS=/data/surgical_dataset/manifests/train.json
export COSMOS_H_SURGICAL_TRANSFER_ENLARGED_FACTORS=1.0
```

Launch the Transfer recipe:

```bash
torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical train \
  --sft-toml examples/post_training/cosmos_h_surgical_transfer_lora_480p.toml \
  2>&1 | tee "$IMAGINAIRE_OUTPUT_ROOT/train_transfer.log"
```

## Export a Training Checkpoint for Inference

Training checkpoints are saved as DCP under the run directory. Select the
latest complete checkpoint and export it back to a Hugging Face safetensors
directory with the resolved training config:

```bash

# Predict run. For Transfer, replace both path components with
# transfer_lora_480p/cosmos_h_surgical_transfer_lora_480p.
RUN_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos_h_surgical/predict_lora_480p/cosmos_h_surgical_predict_lora_480p"
CHECKPOINT_ITER="$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")"
CHECKPOINT_PATH="$RUN_DIR/checkpoints/$CHECKPOINT_ITER"
EXPORT_PATH="$RUN_DIR/model"

python -m cosmos_framework.scripts.export_model \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --config-file "$RUN_DIR/config.yaml" \
  --no-vit \
  --verify \
  -o "$EXPORT_PATH"
```

`--no-vit` is appropriate for this release because the public Predict and
Transfer interfaces are generation-only; it omits the separate reasoner vision
tower and marks the export accordingly. `--verify` runs the pinned framework's
single-GPU generation smoke test after all export artifacts are written. A
failed verification exits nonzero but leaves the export in place for
inspection.

The resulting `$EXPORT_PATH` contains `config.json`, `checkpoint.json`, an
`export_manifest.json`, and one or more safetensor shards with an index. Use it
directly for project inference:

```bash
torchrun --nproc-per-node=8 \
  -m cosmos_h_surgical infer \
  --parallelism-preset=latency \
  --dp-shard-size=1 \
  -i inputs/predict/surgical_predict.jsonl \
  -o outputs/export-verification/predict \
  --checkpoint-path "$EXPORT_PATH" \
  --no-guardrails \
  --seed=0
```

Inspect the exported configuration, metadata, shard inventory, and inference
outputs before uploading the directory to a Hugging Face release branch. The
framework's default `checkpoint.json` records source checkpoint and config
paths for reproducibility; public release automation must replace internal or
user-specific paths with approved portable provenance metadata.

## Checkpoints and Reproducibility

The TOMLs control optimizer, scheduler, output, and checkpoint policy. Before
publishing a trained result, record:

- Base checkpoint revision and digest
- Dataset version, manifest digest, and license
- Complete TOML and source-variable values
- DCP conversion command and `BASE_CHECKPOINT_PATH`
- Cosmos-H-Surgical and Cosmos Framework revisions
- GPU, driver, CUDA, and dependency-lock versions
- Final DCP and exported safetensor inventories and digests

The pinned framework's generic training and TOML references remain useful for
advanced overrides:

- [Post-training](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/training.md)
- [SFT configuration](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/sft_config.md)
