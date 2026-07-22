# Inference Inputs

This directory contains portable Cosmos-H-Surgical v0.3.0 inference inputs for
the two release workflows:

- `predict/`: image-to-video surgical prediction.
- `transfer/`: edge, blur, depth, and segmentation controlled transfer.

The JSON, JSONL, negative prompt, and structured prompt files use paths relative
to their own locations. Run the examples from the repository root with the
commands in each workflow's README.

The local `media/` directories contain validation assets used on DFW. Their
contents are ignored by Git because those source datasets have separate
distribution terms. Only `.gitkeep` is tracked. Before publishing a runnable
example, replace these assets with media cleared for redistribution or publish
them through an approved artifact channel.

See [the inference guide](../docs/inference.md) for the complete interface.
