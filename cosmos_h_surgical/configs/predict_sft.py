# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Nano 480P surgical Predict LoRA recipe."""

from cosmos_framework.utils.lazy_config import LazyCall as L
from hydra.core.config_store import ConfigStore

from cosmos_h_surgical.configs.training_common import (
    common_dataset_kwargs,
    make_packing_dataloader,
    make_surgical_lora_config,
)
from cosmos_h_surgical.data.surgical_video_json_dataset import get_surgical_video_json_dataset

EXPERIMENT_NAME = "cosmos_h_surgical_predict_lora_480p"

cosmos_h_surgical_predict_lora_480p = make_surgical_lora_config(
    experiment_name=EXPERIMENT_NAME,
    group="predict_lora_480p",
)
cosmos_h_surgical_predict_lora_480p.dataloader_train = make_packing_dataloader(
    dataset_name="surgical_predict",
    dataset=L(get_surgical_video_json_dataset)(
        **common_dataset_kwargs(),
        conditioning_config={0: 0.2, 1: 0.7, 2: 0.1},
        dataset_dir="${oc.env:COSMOS_H_SURGICAL_PREDICT_DATASET_DIRS}",
        enlarged_factor="${oc.env:COSMOS_H_SURGICAL_PREDICT_ENLARGED_FACTORS,1.0}",
        json_path="${oc.env:COSMOS_H_SURGICAL_PREDICT_JSON_PATHS}",
    ),
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=cosmos_h_surgical_predict_lora_480p,
)
