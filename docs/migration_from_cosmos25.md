# Migrating from Cosmos 2.5

Cosmos-H-Surgical v0.3.0 moves from two vendored Cosmos 2.5 applications to a
single package over the public Cosmos Framework.

## Archived Release

The complete Cosmos 2.5 release is preserved at:

- Git branch: [`cosmos-2.5`](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/cosmos-2.5)
- Signed tag: [`v0.2.0`](https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical/tree/v0.2.0)
- Hugging Face repository: [`nvidia/Cosmos-H-Surgical`](https://huggingface.co/nvidia/Cosmos-H-Surgical)

Use that branch or tag to reproduce Cosmos 2.5 inference. The Cosmos 3 branch
does not retain the old `predict/` and `transfer/` source trees.

## Repository Change

```text
Cosmos 2.5                         Cosmos 3
-----------                        --------
predict/                            cosmos_h_surgical/
transfer/                           one shared CLI
separate environments               one uv environment
vendored application code           pinned public framework dependency
separate checkpoint families        one mixed-capability checkpoint
```

Local `predict/` or `transfer/` directories may remain after switching an old
working tree to the Cosmos 3 branch because Git does not delete ignored virtual
environments, caches, or outputs. They contain no tracked Cosmos 3 files.

## Command Mapping

Cosmos 2.5 Predict:

```bash
cd predict
python examples/inference.py \
  -i assets/base/coagulation.json \
  -o outputs/predict \
  --inference-type=video2world
```

Cosmos 3 I2V:

```bash
torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical infer \
  -i inputs/i2v.json \
  --output-dir outputs/i2v \
  --checkpoint-path "$COSMOS3_CHECKPOINT" \
  --seed 0
```

Cosmos 2.5 Transfer:

```bash
cd transfer
python examples/inference.py \
  -i assets/example/depth_spec.json \
  -o outputs/depth
```

Cosmos 3 Transfer:

```bash
torchrun --nproc_per_node=8 \
  -m cosmos_h_surgical infer \
  -i inputs/depth_transfer.json \
  --output-dir outputs/depth \
  --checkpoint-path "$COSMOS3_CHECKPOINT" \
  --seed 0 \
  --no-use-torch-compile
```

## Checkpoint Compatibility

Cosmos 2.5 Predict and Transfer checkpoints cannot be loaded by Cosmos 3. The
architectures, configurations, dependencies, and checkpoint layouts differ.
Keep both releases installed separately when reproducing earlier results.

The v0.2.0 license associated with each checkpoint remains unchanged. Cosmos 3
code and the planned v0.3.0 model are released under OpenMDW-1.1 as identified
by the corresponding release manifest.

See [inference.md](inference.md) for Cosmos 3 input and launch details.
