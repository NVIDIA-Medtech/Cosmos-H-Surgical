# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import cosmos_framework
from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L


def _resolve_framework_asset(path: str) -> str:
    """Resolve a framework-repo-relative asset from an installed package."""
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate
    elif candidate.is_file():
        resolved = candidate.resolve()
    else:
        framework_package = Path(cosmos_framework.__file__).resolve().parent
        resolved = framework_package.parent / candidate

    if not resolved.is_file():
        raise FileNotFoundError(f"Unable to resolve packaged Cosmos Framework asset: {path}")
    return str(resolved)


def make_surgical_lora_config(*, experiment_name: str, group: str) -> Any:
    """Return the shared Cosmos3-Nano 480P surgical LoRA configuration."""
    config = copy.deepcopy(vision_sft_nano)
    config.job.project = "cosmos_h_surgical"
    config.job.group = group
    config.job.name = experiment_name

    model_config = config.model.config
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

    vlm_base_config = model_config["vlm_config"]["model_instance"]["config"]["base_config"]
    vlm_base_config["json_file"] = _resolve_framework_asset(str(vlm_base_config["json_file"]))

    config.optimizer.keys_to_select = ["lora_"]
    config.optimizer.lr = 5.0e-4
    config.optimizer.weight_decay = 0.0
    config.scheduler.cycle_lengths = [100000]
    config.scheduler.f_min = [0.1]
    config.scheduler.warm_up_steps = [1000]
    config.trainer.max_iter = 100000
    config.checkpoint.load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
    config.checkpoint.keys_to_skip_loading = ["net_ema."]
    config.checkpoint.save_iter = 50
    config.checkpoint.strict_resume = False
    return config


def make_packing_dataloader(*, dataset_name: str, dataset: Any) -> Any:
    """Wrap one surgical stream in the public framework packing stack."""
    return L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name=dataset_name,
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
            datasets={dataset_name: {"ratio": 1, "dataset": dataset}},
        ),
    )


def common_dataset_kwargs() -> dict[str, Any]:
    """Return settings shared by Predict and Transfer datasets."""
    return {
        "append_duration_fps_timestamps": True,
        "append_resolution_info": True,
        "caption_format": "json",
        "caption_suffix": "",
        "caption_types_and_weights": {"caption_json": 1.0},
        "cfg_dropout_keep_metadata": False,
        "cfg_dropout_rate": 0.1,
        "conditioning_fps": -1,
        "conditioning_fps_noise_std": 0.0,
        "frame_selection_mode": "random",
        "num_frames": 93,
        "temporal_compression_factor": 4,
        "tokenizer_config": "${model.config.vlm_config.tokenizer}",
        "use_system_prompt": False,
        "video_size": (480, 832),
    }
