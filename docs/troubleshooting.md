# Troubleshooting

## Installation

### CUDA dependency groups conflict

Install exactly one CUDA group:

```bash
uv sync --group cu130
# or
uv sync --group cu128
source .venv/bin/activate
```

If the wrong group was installed, remove `.venv` and recreate it with the group
matching the NVIDIA driver. Do not layer both groups into one environment.

### Git LFS files are pointers

Initialize LFS and retrieve the pinned dependency assets:

```bash
git lfs install
git lfs pull
uv sync --group cu130
source .venv/bin/activate
```

### The framework cannot be imported

Run commands from the repository root after `source .venv/bin/activate`.
Confirm that `uv sync` completed and that the framework revision is present in
`uv.lock`.

## Checkpoints

### The default checkpoint cannot be downloaded

Confirm that the `v0.3.0` revision is reachable:

```bash
hf download nvidia/Cosmos-H-Surgical --revision v0.3.0
```

Check network access, Hugging Face authentication, and any
`COSMOS_H_SURGICAL_HF_REPOSITORY` or `COSMOS_H_SURGICAL_HF_REVISION` overrides.
For a local export, pass its directory with `--checkpoint-path`. The directory
must contain `config.json` and all files referenced by that configuration.

### An earlier Cosmos 3 config lacks `enable_input_bias`

The wrapper supports the validated earlier config by creating a temporary
checkpoint view with `enable_input_bias: true`. Existing model files are linked
and the source checkpoint is left unchanged. Ensure the checkpoint parent is
writable enough to create and remove the temporary sibling directory.

## Inputs

### The prompt is plain text

Cosmos-H-Surgical release inference expects `prompt` to contain a serialized
JSON object. Generate one with [prompt_upsampling.md](prompt_upsampling.md) or
provide an equivalent structured prompt manually.

### Both `prompt` and `prompt_path` are set

Pass only one. `prompt_path` should point to a text file containing the
structured JSON prompt.

### A transfer specification is rejected

Confirm that it uses `model_mode: "video2video"`, supplies exactly one control,
and provides all paths required by that control. Valid release controls are
`edge`, `blur`, `depth`, and `seg`.

`resize_mode` accepts only `preserve_aspect` and `stretch`.

## Distributed Inference

### NCCL watchdog timeout

The wrapper defaults the NCCL timeout to 1,800 seconds. Set a larger positive
integer only when startup is legitimately slow:

```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400
```

A timeout can also indicate mismatched collective execution, an invalid GPU
topology, or a stalled process; increasing it is not a general fix.

### Transfer stalls before sampling

Pass the Transfer specifications to one `torchrun` and disable compilation:

```bash
torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical infer \
  --parallelism-preset=latency \
  --dp-shard-size=1 \
  -i "inputs/transfer/specs/*.json" \
  --output-dir outputs/transfer \
  --seed 0 \
  --no-use-torch-compile \
  --no-guardrails
```

Keep the glob quoted. The wrapper expands it deterministically, keeps the model
loaded once, and serializes Transfer samples internally so distributed
collectives remain aligned. If a stall persists, retry one specification to
isolate the failing input or control.

### Address already in use

Choose another rendezvous port:

```bash
torchrun --master-port 29502 --nproc_per_node=8 \
  -m cosmos_h_surgical infer --help
```

### CUDA out of memory

Stop other GPU processes, verify the requested process count, and inspect the
framework's parallelism arguments. Compilation can consume additional memory;
retry with `--no-use-torch-compile` when diagnosing the failure.

## Prompt Upsampling

### Endpoint, model, or token is missing

Set `PROMPT_UPSAMPLER_ENDPOINT_URL`, `PROMPT_UPSAMPLER_MODEL`, and
`PROMPT_UPSAMPLER_API_TOKEN`, or pass the corresponding CLI options. Confirm
that the endpoint follows an OpenAI-compatible request contract.

### Output is not structured JSON

Inspect `prompt_<index>.json` and parse the string under `prompt` as JSON. Retry
with a compatible model/template when the endpoint returns prose, Markdown, or
an incomplete object.

## Reporting a Reproducible Failure

Include the repository commit, `framework-info`, checkpoint revision, input
specification, CUDA group, GPU type/count, exact command, and the complete
`sample_outputs.json`. Remove credentials and private paths before sharing.
