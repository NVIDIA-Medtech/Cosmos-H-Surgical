# Inference

Cosmos-H-Surgical focuses on two validated surgical video workflows:

- (Predict) Image-to-video prediction from a surgical starting frame
- (Transfer) Video transfer conditioned by edge, depth, segmentation, or blur controls

The package forwards inference to the pinned Cosmos Framework CLI while adding
release-checkpoint and surgical transfer compatibility. It selects the
`Cosmos-H-Surgical` checkpoint by default.

## Structured Prompts Are Required

Release inference uses structured JSON prompts. There are two JSON layers:

1. The outer JSON or JSONL record contains inference arguments.
2. The `prompt` value is a serialized JSON object describing the scene and
   temporal evolution.

An abbreviated input looks like this:

```json
{
  "name": "needle_transfer",
  "model_mode": "image2video",
  "prompt": "{\"subjects\":[{\"description\":\"Two robotic needle drivers\",\"action\":\"The right driver passes a curved needle to the left driver\"}],\"background_setting\":\"Close laparoscopic view of soft tissue\",\"actions\":[{\"time\":\"0:00-0:03\",\"description\":\"The right driver approaches the left driver with the needle\"},{\"time\":\"0:03-0:06\",\"description\":\"The left driver grasps the needle and the right driver releases it\"}],\"temporal_caption\":\"The right needle driver passes the curved needle to the left needle driver.\"}",
  "vision_path": "inputs/needle_transfer.png",
  "resolution": "480",
  "aspect_ratio": "16,9",
  "fps": 16,
  "num_frames": 93,
  "guidance": 6.0,
  "shift": 5.0
}
```

The structured prompt may also be placed in a text file and selected with
`prompt_path`. Pass only one of `prompt` and `prompt_path`. See
[prompt_upsampling.md](prompt_upsampling.md) for generating structured prompts
from short natural-language descriptions.

## Common Input Fields

| Field | Purpose |
| --- | --- |
| `name` | Unique sample name and output subdirectory. |
| `model_mode` | `image2video` for prediction or `video2video` for transfer. |
| `prompt` | Serialized structured JSON prompt. |
| `prompt_path` | Alternative file containing the structured prompt. |
| `vision_path` | Starting image or source video, depending on the mode. |
| `resolution` | Resolution tier; the release candidate uses `480`. |
| `aspect_ratio` | Aspect-ratio key; the release candidate uses `16,9`. |
| `fps` | Output frame rate; validated at 16 FPS. |
| `num_frames` | Output frame count; validated at 93 frames. |
| `seed` | Per-sample random seed. A CLI `--seed` override takes precedence. |
| `guidance` | Text-guidance strength. |
| `shift` | Diffusion time-shift value. |

JSONL files contain one complete outer record per line. JSON files contain one
record. Paths inside a record may be absolute or relative to the input file.
Do not publish specifications containing private storage paths.

## Checkpoint

No checkpoint option is required for the public release:

```bash
uv run --no-sync cosmos-h-surgical infer --help
```

The wrapper injects `--checkpoint-path Cosmos-H-Surgical` when the option is
omitted. That registered name resolves to `nvidia/Cosmos-H-Surgical`. The release manifest records the model name, revision, and
SHA-256 digest. Cosmos 2.5 checkpoints are not compatible with this loader and
remain available from the `cosmos-2.5` release branch.

To test a local export, provide an explicit path:

```bash
--checkpoint-path /path/to/cosmos-h-surgical-local-checkpoint-path
```

To test a private release candidate while retaining the public model name, set
`COSMOS_H_SURGICAL_HF_REPOSITORY` and `COSMOS_H_SURGICAL_HF_REVISION`. Explicit
checkpoint options always take precedence over the default.

The wrapper may create a temporary sibling view when loading an earlier Cosmos
3 checkpoint configuration. It links existing model files and writes only the
required compatibility configuration into the temporary directory. It does
not modify the source checkpoint.

## Image-to-Video Prediction

Save one or more I2V records in `inputs/i2v.json` or `inputs/i2v.jsonl`. Run the
validated eight-GPU configuration with:

```bash
uv run --no-sync torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical infer \
  -i inputs/i2v.json \
  --output-dir outputs/i2v \
  --seed 0
```

The CLI accepts multiple input files and glob patterns after `-i`. Batch I2V
was validated from JSONL with the model loaded once for all samples.

## Transfer Inference

Transfer uses `model_mode: "video2video"` and exactly one active control in each
specification. All four release controls use structured prompts.

| Control | Input |
| --- | --- |
| Edge | A source video in `vision_path`; Canny hints are generated using the selected preset. |
| Blur | A source video in `vision_path`; blur hints are generated using the selected preset. |
| Depth | A depth-control video in `depth.control_path`. |
| Segmentation | A segmentation-control video in `seg.control_path`. |

Common transfer fields include:

```json
{
  "name": "transfer_example",
  "model_mode": "video2video",
  "prompt_path": "prompts/transfer_example.json",
  "resolution": "480",
  "aspect_ratio": "16,9",
  "num_frames": 93,
  "fps": 16,
  "num_steps": 50,
  "seed": 0,
  "guidance": 1.0,
  "control_guidance": 1.0,
  "shift": 5.0,
  "resize_mode": "stretch"
}
```

### Edge

```json
{
  "vision_path": "inputs/source.mp4",
  "edge": {
    "preset_edge_threshold": "medium"
  }
}
```

### Blur

```json
{
  "vision_path": "inputs/source.mp4",
  "blur": {
    "preset_blur_strength": "medium"
  }
}
```

### Depth

```json
{
  "depth": {
    "control_path": "inputs/source.depth.mp4"
  }
}
```

### Segmentation

```json
{
  "seg": {
    "control_path": "inputs/source.seg.mp4"
  }
}
```

The fragments above are additions to the common transfer record; they are not
standalone input files.

### Resize Modes

`preserve_aspect` uses the framework's aspect-preserving preprocessing.
`stretch` resizes the control directly to the requested output dimensions. The
wrapper accepts the release-candidate `resize_mode` field and passes its value
through a process-local compatibility adapter.

### Recommended Launch Pattern

Pass one or more transfer specifications to a single `torchrun`. The release
wrapper expands quoted glob patterns and processes transfer specifications in
deterministic order while keeping the model loaded once. The validated path
disables `torch.compile`:

```bash
uv run --no-sync torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical infer \
  --parallelism-preset=latency \
  --dp-shard-size=1 \
  --no-use-torch-compile \
  -i "inputs/transfer/specs/*.json" \
  --output-dir outputs/transfer \
  --seed 0 \
  --no-guardrails
```

Keep the glob quoted so the wrapper receives and expands it consistently. Each
transfer sample uses the framework's single-spec distributed batching path;
this keeps control-specific collectives aligned across ranks. I2V JSONL and
multi-file batching continue to use the framework's regular batch scheduler.

## Parallelism and Performance

The framework defaults to the `latency` parallelism preset and enables
`torch.compile`. Inspect all available options with:

```bash
uv run --no-sync cosmos-h-surgical infer --help
```

Useful options include:

```text
--parallelism-preset {throughput,latency}
--cp-size N
--cfgp-size N
--use-torch-compile / --no-use-torch-compile
--use-cuda-graphs / --no-use-cuda-graphs
```

The wrapper sets the distributed timeout to 1,800 seconds by default. Override
it before launch only when necessary:

```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
```

## Reproducibility

Use an explicit seed for every release comparison. A fixed seed controls the
initial random state, but results can still vary with GPU architecture, CUDA
libraries, parallelism, compilation, or prompt changes. Record the following
with any reported result:

- Cosmos-H-Surgical commit or tag
- Cosmos Framework revision from `framework-info`
- Checkpoint revision and digest
- Input specification
- Seed
- CUDA dependency group
- GPU model and count
- Compilation and parallelism settings

## Outputs

Each sample is written below `<output-dir>/<name>/`. Typical artifacts include:

```text
<output-dir>/<name>/
|-- inputs/
|-- sample_args.json
|-- sample_outputs.json
`-- vision.mp4
```

Transfer runs may also save the generated control hint when
`show_control_condition` is enabled. Treat `sample_outputs.json` as the status
record: a successful sample reports `status: "success"`.

See [troubleshooting.md](troubleshooting.md) for checkpoint, CUDA, NCCL, and
input-validation failures. The complete generic interface is documented in the
[pinned Cosmos Framework inference guide](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/inference.md).
