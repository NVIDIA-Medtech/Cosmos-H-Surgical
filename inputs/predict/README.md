# Surgical Prediction Inputs

`surgical_predict.jsonl` contains ten 480p image-to-video release-review
records: aspiration, coagulation, dissection, knotting, needle puncture, suture
pulling, needle grasping, clipping, packing, and tissue retraction.
It uses structured prompts and action-named source images from `media/`.

From the repository root:

```bash
torchrun --nproc-per-node=8 \
  -m cosmos_h_surgical infer \
  --parallelism-preset=latency \
  --dp-shard-size=1 \
  -i inputs/predict/surgical_predict.json \
  -o outputs/predict-smoke \
  --no-guardrails \
  --seed=0
```
