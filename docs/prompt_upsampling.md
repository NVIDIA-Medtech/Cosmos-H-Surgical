# Prompt Upsampling

Cosmos-H-Surgical inference expects a structured JSON prompt. The prompt
upsampler in the pinned Cosmos Framework converts short natural-language
descriptions into that format through an OpenAI-compatible API endpoint.

This guide covers the image- and video-conditioned modes relevant to
Cosmos-H-Surgical. It is adapted from the
[prompt-upsampling guide at the pinned framework revision](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/docs/prompt_upsampling.md).

## Configure an Endpoint

Set the endpoint, model, and token expected by your OpenAI-compatible service:

```bash
export PROMPT_UPSAMPLER_ENDPOINT_URL="https://example.invalid/v1/"
export PROMPT_UPSAMPLER_MODEL="model-name"
export PROMPT_UPSAMPLER_API_TOKEN="token"
```

The module reads `PROMPT_UPSAMPLER_MODEL`. Do not use
`PROMPT_UPSAMPLER_MODEL_NAME`; that name is not consumed by the pinned CLI.
Keep API tokens in the environment and never place them in input files or Git.

Optional template overrides are:

```bash
export PROMPT_UPSAMPLER_PROMPT_TEMPLATE=/path/to/prompt_template.txt
export PROMPT_UPSAMPLER_JSON_TEMPLATE=/path/to/json_schema.json
```

When omitted, the built-in templates from the pinned framework are used.

### Default I2V Templates

The default `image2video` workflow uses these templates from the pinned Cosmos
Framework revision:

- [`t2v_i2v_video_prompt.txt`](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/cosmos_framework/inference/prompting_templates/external_api/t2v_i2v_video_prompt.txt)
  instructs the endpoint to expand the short prompt and input image into a
  dense, physically plausible video description. It also injects the requested
  resolution, aspect ratio, duration, FPS, and JSON schema.
- [`t2v_i2v_video_json_schema.json`](https://github.com/NVIDIA/cosmos-framework/blob/ed8287fd7477113f8ac4f6b84290514d55cf0cdc/cosmos_framework/inference/prompting_templates/external_api/t2v_i2v_video_json_schema.json)
  defines the exact structured output fields, including subjects, scene,
  lighting, cinematography, timed actions, segments, temporal caption, and
  video metadata.

Use the override variables above only when a compatible custom prompt and
schema have been validated together.

## Prepare Inputs

This repository provides two aligned input lists:

- `inputs/surgical_prompts.txt`: one short surgical prompt per non-empty line.
- `inputs/surgical_images.txt`: one matching image path per non-empty line.

The prompt list covers the ten release-review actions and begins with:

```text
real surgery scene: left fenestrated bipolar forceps holds tissue while right suction-irrigator aspirates blood on tissue
real surgery scene: right instrument coagulates the cystic mesentery while  left instrument retracts the cystic mesentery
```

The image list uses the corresponding action-named media already included under
`inputs/predict/media/`:

```text
inputs/predict/media/aspiration.jpg
inputs/predict/media/coagulation.jpg
```

Run prompt upsampling from the repository root so these paths resolve as
written. For a custom I2V run, pass either one shared image:

```text
--image-url inputs/predict/media/aspiration.jpg
```

or use the provided file containing one image path per prompt:

```text
--image-list inputs/surgical_images.txt
```

The image list and prompt file must contain the same number of non-empty lines.
Local paths, HTTP(S) URLs, and `data:` URLs are accepted by the upstream tool.

For video-level prompt upsampling, the repository also provides two aligned
example lists without source case identifiers:

- `inputs/video_prompts.txt`: short coagulation and needle-grasping prompts.
- `inputs/surgical_videos.txt`: matching local videos from
  `inputs/transfer/media/`.

## Generate Structured Prompts

Use `image2video` for the general Cosmos 3 I2V schema:

```bash
uv run --no-sync python -m cosmos_h_surgical prompt-upsample \
  --input inputs/surgical_prompts.txt \
  --image-list inputs/surgical_images.txt \
  --output outputs/upsampled-prompts \
  --mode image2video \
  --resolution 480 \
  --aspect-ratio "16,9" \
  --duration "6s" \
  --fps 16 \
  --verbose
```

The Cosmos-H-Surgical wrapper omits `top_k` by default because some
OpenAI-compatible APIs reject that provider-specific sampling field. Pass an
integer such as `--top-k 20` only when the selected endpoint supports it;
`--top-k none` explicitly keeps it omitted.

Normal output reports overall item progress and writes each completed
`prompt_<index>.json` atomically. If a later request fails, earlier completed
files remain usable. `--verbose` additionally reports sanitized HTTP request
and response status, request latency, retry attempts, and sampled timestamps;
credentials and request bodies are never logged.

### Video-Level Prompt Upsampling

Use `video` mode when the prompt upsampler should observe temporal changes
across a source video instead of receiving only its first frame:

```bash
uv run --no-sync python -m cosmos_h_surgical prompt-upsample \
  --input inputs/video_prompts.txt \
  --video-list inputs/surgical_videos.txt \
  --output outputs/upsampled-video-prompts \
  --mode video \
  --resolution 480 \
  --aspect-ratio "16,9" \
  --duration "6s" \
  --fps 16 \
  --top-k none \
  --verbose
```

The default sampler extracts at least 10 chronological frames and maintains a
minimum temporal density of 1 frame per second over the complete source video:

```text
sample_count = max(10, ceil(source_duration_seconds * 1.0))
```

Frames are decoded on CPU, uniformly distributed over temporal bins, resized
only when an edge exceeds 768 pixels, and encoded as JPEG data URLs. The
endpoint receives timestamp labels and explicit instructions to treat the
frames as one chronological sequence. Requests are limited to 32 frames by
default; a longer video that requires more frames fails clearly instead of
silently violating the minimum sampling rate.

These defaults can be changed with `--video-frame-count`, `--video-min-fps`,
and `--video-max-frames`. Use `--video PATH` for one shared local video or
`--video-list PATH` for one local video per prompt. Video mode does not accept
remote URLs because frames must be decoded locally before the API request.

The command writes one `prompt_<index>.json` file per input line. Each file
contains a compact serialized JSON object under `prompt`:

```json
{
  "prompt": "{\"subjects\":[...],\"actions\":[...],\"temporal_caption\":\"...\"}"
}
```

## Build an Inference Specification

Prompt-upsampler output is a prompt record, not a complete inference record.
Add the sample name, modality, image, video settings, and inference parameters:

```json
{
  "name": "aspiration",
  "model_mode": "image2video",
  "prompt": "{\"subjects\":[...],\"actions\":[...],\"temporal_caption\":\"...\"}",
  "vision_path": "inputs/predict/media/aspiration.jpg",
  "resolution": "480",
  "aspect_ratio": "16,9",
  "fps": 16,
  "num_frames": 93,
  "guidance": 6.0,
  "shift": 5.0
}
```

Alternatively, copy the serialized structured object into a prompt text file
and use `prompt_path` from the complete inference record.

## Validate Before Inference

Before launching the model, verify that:

1. The outer prompt-upsample output is valid JSON.
2. Its `prompt` value parses as a JSON object, not plain prose.
3. Every sample has the required image or source-video path and video parameters.
4. The prompt describes visible surgical subjects, actions, scene context, and
   temporal evolution.
5. No endpoint credentials or private storage locations appear in files that
   will be published.

Then run the completed specification as described in
[inference.md](inference.md).
