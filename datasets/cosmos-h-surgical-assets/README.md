# Cosmos-H-Surgical Assets

This directory contains a small, training-ready synthetic surgical video
dataset for the Cosmos-H-Surgical Predict and Transfer post-training examples.
All ten RGB target videos are synthetic data selected from the corresponding
action examples in the synthetic training corpus. Public filenames expose only
the action label. Aligned caption, blur, depth, and segmentation sidecars
support the repository's training data contract.

## Layout

Each target has adjacent caption and control files with the same stem:

```text
videos/
|-- aspiration.mp4
|-- aspiration.json
|-- aspiration.blur.mp4
|-- aspiration.depth.mp4
`-- aspiration.seg.mp4
```

Caption files contain only the structured prompt consumed by the training
loader:

```json
{
  "caption_json": {
    "description": "A structured description of the surgical clip."
  }
}
```

Blur, depth, and segmentation are materialized as aligned videos. Edge control
is computed from the RGB target at load time and therefore has no `.edge.mp4`
sidecar.

The portable training split is defined in `manifests/train.json`. All media
files are managed with Git LFS; JSON and documentation remain regular Git
files.

## Validation

From the repository root, validate the same manifest for both recipes:

```bash
cosmos-h-surgical validate-training-data \
  --mode predict \
  --dataset-dir datasets/cosmos-h-surgical-assets \
  --manifest datasets/cosmos-h-surgical-assets/manifests/train.json

cosmos-h-surgical validate-training-data \
  --mode transfer \
  --dataset-dir datasets/cosmos-h-surgical-assets \
  --manifest datasets/cosmos-h-surgical-assets/manifests/train.json \
  --control-modalities edge,blur,depth,seg
```

See the repository `LICENSE` and `NOTICE` files for applicable terms and
attribution information.
