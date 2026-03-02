# export HF_HOME=/workspace/code/cosmos-transfer2.5/cache
export HF_HOME=/workspace/code/Cosmos-H-Surgical-gitlab/transfer/cache
export IMAGINAIRE_OUTPUT_ROOT=/workspace/code/cosmos-transfer2.5/imaginaire4-output


# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i /healthcareeng_monai/datasets/BSA_HighFPS/bsa_10case_val_split_short_prompt_transfer_seg.jsonl \
#     -o outputs/cosmos_surgical_singleview_posttrain_seg_v2_20000s_test_hf


# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i /healthcareeng_monai/datasets/medbot_1027/medbot_45_cases_short_prompt_transfer_seg_guided_edited.jsonl \
#     -o outputs/medbot_45_cases_short_prompt_transfer_seg_guided_edited3_test_hf

# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i assets/depth.jsonl \
#     -o outputs/depth_test_hf

# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i assets/seg.jsonl \
#     -o outputs/seg_test_hf

# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i assets/edge.jsonl \
#     -o outputs/edge_test_hf

# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i assets/vis.jsonl \
#     -o outputs/vis_test_hf

# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i assets/multicontrol.jsonl \
#     -o outputs/multicontrol_test_hf

# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i assets/needlePuncture_guided_example/seg/needlePuncture_guided_seg_spec.json \
#     -o outputs/needlePuncture_guided_seg_test_hf

# torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
#     -i assets/needleGrasping_guided_example/seg/needleGrasping_guided_seg_spec.json \
#     -o outputs/needleGrasping_guided_seg_test_hf

torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
    -i assets/image_example/image2image.json \
    -o outputs/image2image_test_hf

torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
    -i assets/image_example/image2image_guided.json \
    -o outputs/image2image_guided_test_hf