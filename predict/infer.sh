#! /bin/bash
# export HF_HOME=/workspace/code/cosmos-predict2.5/cache
# torchrun --nproc_per_node=8 examples/inference.py -i assets/base/robot_pouring.json -o outputs/base_video2world6 --inference-type=video2world

CHECKPOINT=/workspace/code/cosmos-predict2.5/imaginaire4-output/cosmos_predict_v2p5/video2world/2b_bsa_jhu_grasp_v2_sft_iter_000038000_bk/model_ema_bf16.pt
OUTPUT_DIR=outputs/2b_bsa_jhu_grasp_v2_sft_38000s_4
INPUT=/healthcareeng_monai/datasets/BSA_HighFPS/bsa_onccase_val_split_short_prompt.jsonl

# export HF_HOME=/workspace/code/Cosmos-H-Surgical-gitlab/predict/cache
# export IMAGINAIRE_OUTPUT_ROOT=/workspace/code/Cosmos-H-Surgical-gitlab/predict/imaginaire4-output
# torchrun --nproc_per_node=8 examples/inference.py \
#   -i $INPUT \
#   -o $OUTPUT_DIR

export HF_HOME=/workspace/code/Cosmos-H-Surgical-gitlab/predict/cache
export IMAGINAIRE_OUTPUT_ROOT=/workspace/code/Cosmos-H-Surgical-gitlab/predict/imaginaire4-output

# OUTPUT_DIR=outputs/batch_input_image2world
# INPUT=assets/batch_input.jsonl
# torchrun --nproc_per_node=8 examples/inference.py \
#   -i $INPUT \
#   -o $OUTPUT_DIR

OUTPUT_DIR=outputs/aspiration_long_image2world
INPUT=assets/base/aspiration_long.json
torchrun --nproc_per_node=8 examples/inference.py \
  -i $INPUT \
  -o $OUTPUT_DIR