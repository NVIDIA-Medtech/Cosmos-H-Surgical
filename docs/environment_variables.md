# Environment Variables

Cosmos-H-Surgical uses a small set of project variables plus variables
inherited from the pinned Cosmos Framework and its dependencies.

## Inference and Storage

| Variable | Purpose |
| --- | --- |
| `HF_HOME` | Hugging Face cache root for downloaded models and tokenizers. |
| `HF_TOKEN` | Optional Hugging Face token for gated artifacts or higher request limits. |
| `COSMOS_H_SURGICAL_HF_REPOSITORY` | Override the default `nvidia/Cosmos-H-Surgical` repository for inference and post-training. |
| `COSMOS_H_SURGICAL_HF_REVISION` | Override the default `v0.3.0` model revision for inference and post-training. |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | Distributed timeout in seconds; defaults to `1800` in the surgical wrapper. |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA device selection. |
| `MASTER_ADDR` | Distributed launch coordinator address. |
| `MASTER_PORT` | Distributed launch coordinator port. |
| `TMPDIR` | Temporary-file root used by Python and dependencies. |

`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` must be a positive integer. The wrapper also
applies the same value to PyTorch's default NCCL process-group timeout before
the framework creates its distributed mesh.

## Prompt Upsampling

| Variable | Purpose |
| --- | --- |
| `PROMPT_UPSAMPLER_ENDPOINT_URL` | OpenAI-compatible API endpoint. |
| `PROMPT_UPSAMPLER_MODEL` | Model identifier served by the endpoint. |
| `PROMPT_UPSAMPLER_API_TOKEN` | Endpoint API token. |
| `PROMPT_UPSAMPLER_PROMPT_TEMPLATE` | Optional prompt-template override. |
| `PROMPT_UPSAMPLER_JSON_TEMPLATE` | Optional JSON-schema-template override. |
| `JSON_ENSURE_ASCII` | Set to `0` to preserve non-ASCII characters in prompt-upsample output. |

The pinned framework reads `PROMPT_UPSAMPLER_MODEL`, not
`PROMPT_UPSAMPLER_MODEL_NAME`.

## Post-Training

The included surgical LoRA recipe uses:

| Variable | Purpose |
| --- | --- |
| `COSMOS_H_SURGICAL_PREDICT_DATASET_DIRS` | Comma-separated roots containing Predict target videos and caption sidecars. |
| `COSMOS_H_SURGICAL_PREDICT_JSON_PATHS` | Comma-separated Predict manifest JSON files paired with the dataset roots. |
| `COSMOS_H_SURGICAL_PREDICT_ENLARGED_FACTORS` | Optional comma-separated Predict repeat/subsample factors; defaults to `1.0`. |
| `COSMOS_H_SURGICAL_TRANSFER_DATASET_DIRS` | Comma-separated roots containing Transfer targets, captions, and controls. |
| `COSMOS_H_SURGICAL_TRANSFER_JSON_PATHS` | Comma-separated Transfer manifest JSON files paired with the dataset roots. |
| `COSMOS_H_SURGICAL_TRANSFER_ENLARGED_FACTORS` | Optional comma-separated Transfer repeat/subsample factors; defaults to `1.0`. |
| `BASE_CHECKPOINT_PATH` | Shared path to the released Cosmos-H-Surgical checkpoint after conversion to DCP. |
| `WAN_VAE_PATH` | Resolved local path to `Wan2.2_VAE.pth`, normally returned by `hf download` under `HF_HOME`. |

Download the released safetensors using the repository and revision variables
above, then convert that complete snapshot once with
`cosmos_framework.scripts.convert_model_to_dcp`. New post-training runs load
the resulting `BASE_CHECKPOINT_PATH`; subsequent run checkpoints remain DCP
until explicitly exported back to Hugging Face safetensors. See
[post_training.md](post_training.md) for the conversion and export commands.

## Example

```bash
export HF_HOME=/path/to/huggingface-cache
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export MASTER_PORT=29501
```

To test an alternate repository or private revision:

```bash
export COSMOS_H_SURGICAL_HF_REPOSITORY=your-org/Cosmos-H-Surgical-private
export COSMOS_H_SURGICAL_HF_REVISION=your-revision
```

For prompt upsampling:

```bash
export PROMPT_UPSAMPLER_ENDPOINT_URL="https://example.invalid/v1/"
export PROMPT_UPSAMPLER_MODEL="model-name"
export PROMPT_UPSAMPLER_API_TOKEN="token"
```

Never commit tokens, credentials, private endpoint URLs, or internal storage
paths. Environment variables beginning with `COSMOS_TRAINING` are managed
internally by the wrapper and should not be set by users.
