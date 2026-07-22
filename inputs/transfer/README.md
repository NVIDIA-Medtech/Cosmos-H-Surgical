# Surgical Transfer Inputs

This bundle contains eight 480p transfer examples. Coagulation and needle
grasping each have independent blur, segmentation, edge, and depth controls.
Every specification activates exactly one control, uses a structured prompt
from `prompts/`, and resolves action-named media from `media/`.

Run all specifications from the repository root:

```bash
torchrun --nproc-per-node=8 \
  -m cosmos_h_surgical infer \
  --parallelism-preset=latency \
  --dp-shard-size=1 \
  --no-use-torch-compile \
  -i "inputs/transfer/specs/*.json" \
  -o outputs/transfer \
  --no-guardrails \
  --seed=0
```

Keep the glob quoted. The Cosmos-H-Surgical wrapper expands it and processes
the transfer specifications sequentially while loading the model only once.
