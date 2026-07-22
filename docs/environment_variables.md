# Environment Variables

Cosmos-H-Surgical uses a small set of project variables plus variables
inherited from the pinned Cosmos Framework and its dependencies.

## Inference and Storage

| Variable | Purpose |
| --- | --- |
| `HF_HOME` | Hugging Face cache root for downloaded models and tokenizers. |
| `HF_TOKEN` | Optional Hugging Face token for gated artifacts or higher request limits. |
| `COSMOS_H_SURGICAL_HF_REPOSITORY` | Override the default `nvidia/Cosmos-H-Surgical` repository, primarily for private RC validation. |
| `COSMOS_H_SURGICAL_HF_REVISION` | Override the default `v0.3.0` model revision. |
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

## Post-Training Preview

The included surgical LoRA recipe uses:

| Variable | Purpose |
| --- | --- |
| `COSMOS_H_SURGICAL_DATASET` | Absolute path to the surgical training JSONL. |
| `BASE_CHECKPOINT_PATH` | Base Cosmos 3 checkpoint used to start training. |
| `WAN_VAE_PATH` | Wan VAE checkpoint required by the recipe. |

See [post_training.md](post_training.md) before using these variables.

## Example

```bash
export HF_HOME=/path/to/huggingface-cache
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export MASTER_PORT=29501
```

For the private v0.3.0 release candidate:

```bash
export COSMOS_H_SURGICAL_HF_REPOSITORY=pengfeig/Cosmos-H-Surgical-staging
export COSMOS_H_SURGICAL_HF_REVISION=rc/v0.3.0-cosmos3
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
