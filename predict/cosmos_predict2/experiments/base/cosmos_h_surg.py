# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey
from cosmos_predict2.experiments.base.cosmos_h_surg_lora import dataloader_train_cosmos_h_surgical_json

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

cosmos_h_surgical_predict_image2world_2b_sft = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_h_surgical_predict",
        group="image2world",
        name="cosmos_h_surgical_predict_image2world_2b_sft",
    ),
    dataloader_train=dataloader_train_cosmos_h_surgical_json,
    checkpoint=dict(
        save_iter=50,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
        verbosity_interval=5,
    ),
    trainer=dict(
        logging_iter=5,
        max_iter=100_000,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=100,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=200_000,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=200_000,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
        ),
    ),
    model=dict(
        config=dict(
            # Training configuration for all three modes
            # The model will randomly sample between 0, 1, and 2 conditional frames during training
            min_num_conditional_frames=0,  # Allow text2world (0 frames)
            max_num_conditional_frames=2,  # Allow up to video2world (2 frames)
            # Probability distribution for sampling number of conditional frames
            # This controls how often each mode is trained:
            # - 0 frames: text2world (33.3%)
            # - 1 frame: image2world (33.3%)
            # - 2 frames: video2world (33.3%)
            conditional_frames_probs={0: 0.0, 1: 1.0, 2: 0.0},
            # Optional: set conditional_frame_timestep for better control
            conditional_frame_timestep=-1.0,  # Default -1 means not effective
            # Keep the default conditioning strategy
            conditioning_strategy="frame_replace",
            denoise_replace_gt_frames=True,
        ),
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
)

cs = ConfigStore.instance()

# Register the configuration with Hydra ConfigStore
for _item in [
    cosmos_h_surgical_predict_image2world_2b_sft,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
