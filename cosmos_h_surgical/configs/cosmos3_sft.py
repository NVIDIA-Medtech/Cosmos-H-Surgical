# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Public Cosmos 3 Nano 480P surgical vision LoRA recipe.

The first public migration covers T2V plus first-frame/short-continuation I2V.
Mixed transfer and action streams will be registered separately once their
dataset adapters and schemas are part of the public package.
"""

import copy

from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.data.generator.local_datasets.sft_dataset import get_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L
from hydra.core.config_store import ConfigStore

EXPERIMENT_NAME = "cosmos_h_surgical_vision_lora_480p"

cosmos_h_surgical_vision_lora_480p = copy.deepcopy(vision_sft_nano)
cosmos_h_surgical_vision_lora_480p.job.project = "cosmos_h_surgical"
cosmos_h_surgical_vision_lora_480p.job.group = "vision_lora_480p"
cosmos_h_surgical_vision_lora_480p.job.name = EXPERIMENT_NAME

model_config = cosmos_h_surgical_vision_lora_480p.model.config
model_config["action_gen"] = False
model_config["vision_gen"] = True
model_config["sound_gen"] = False
model_config["max_action_dim"] = 32
model_config["max_num_tokens_after_packing"] = 45056
model_config["resolution"] = "480"
model_config["lora_enabled"] = True
model_config["lora_rank"] = 16
model_config["lora_alpha"] = 32
model_config["lora_target_modules"] = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
model_config["ema"]["enabled"] = False
model_config["parallelism"]["data_parallel_shard_degree"] = 8
model_config["rectified_flow_training_config"]["shift"]["480"] = 5
model_config["tokenizer"]["chunk_duration"] = 93

cosmos_h_surgical_vision_lora_480p.optimizer.keys_to_select = ["lora_"]
cosmos_h_surgical_vision_lora_480p.optimizer.lr = 5.0e-4
cosmos_h_surgical_vision_lora_480p.optimizer.weight_decay = 0.0
cosmos_h_surgical_vision_lora_480p.scheduler.cycle_lengths = [100000]
cosmos_h_surgical_vision_lora_480p.scheduler.f_min = [0.1]
cosmos_h_surgical_vision_lora_480p.scheduler.warm_up_steps = [1000]
cosmos_h_surgical_vision_lora_480p.trainer.max_iter = 100000
cosmos_h_surgical_vision_lora_480p.checkpoint.keys_to_skip_loading = ["net_ema.", "lora_"]
cosmos_h_surgical_vision_lora_480p.checkpoint.save_iter = 50
cosmos_h_surgical_vision_lora_480p.checkpoint.strict_resume = False

cosmos_h_surgical_vision_lora_480p.dataloader_train = L(PackingDataLoader)(
    audio_sample_rate=48000,
    dataset_name="surgical_video",
    max_samples_per_batch=None,
    max_sequence_length=45056,
    patch_spatial=2,
    sound_latent_fps=0,
    tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4,
    dataloader=L(RankPartitionedDataLoader)(
        batch_size=1,
        in_order=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
        sampler=None,
        datasets=dict(
            video=dict(
                ratio=1,
                dataset=L(get_sft_dataset)(
                    append_duration_fps_timestamps=True,
                    append_resolution_info=True,
                    max_caption_tokens=2048,
                    caption_suffix="",
                    cfg_dropout_keep_metadata=False,
                    cfg_dropout_rate=0.1,
                    conditioning_config={0: 0.2, 1: 0.7, 2: 0.1},
                    conditioning_fps=-1,
                    conditioning_fps_noise_std=0.0,
                    frame_selection_mode="random",
                    jsonl_paths=["${oc.env:COSMOS_H_SURGICAL_DATASET}"],
                    min_short_edge=480,
                    num_video_frames=93,
                    resolution="480",
                    sample_by_window=False,
                    temporal_compression_factor=4,
                    temporal_interval_mode="max_30fps",
                    use_system_prompt=False,
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                ),
            ),
        ),
    ),
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=cosmos_h_surgical_vision_lora_480p,
)
