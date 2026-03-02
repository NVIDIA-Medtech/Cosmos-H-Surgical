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

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    get_generic_dataloader,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video_json import (
    VideoDatasetJSON,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# Example datasets
# JSON format:
# {
#     "training": [
#         {
#             "video": "video1.mp4",
#         },
#         {
#             "video": "video2.mp4",
#         }
#         {
#             "video": "video3.mp4",
#         }
#     ]
# }
DATA_ROOT_1 = "/datasets/dataset1"
DATA_LIST_1 = "/datasets/dataset1/dataset1.json"

DATA_ROOT_2 = "/datasets/dataset2"
DATA_LIST_2 = "/datasets/dataset2/dataset2.json"

DATA_ROOT_3 = "/datasets/dataset3"
DATA_LIST_3 = "/datasets/dataset3/dataset3.json"

DATASET_DIR = ",".join(
    [
        DATA_ROOT_1,
        DATA_ROOT_2,
        DATA_ROOT_3,
    ]
)

JSON_PATH = ",".join(
    [
        DATA_LIST_1,
        DATA_LIST_2,
        DATA_LIST_3,
    ]
)
ENLARGED_FACTOR = "1.0,1.0,1.0"

video_dataset_cosmos_h_surgical_json = L(VideoDatasetJSON)(
    dataset_dir=DATASET_DIR,
    json_path=JSON_PATH,
    enlarged_factor=ENLARGED_FACTOR,
    num_frames=93,
    video_size=(704, 1280),
    caption_format="json",
    prompt_type="short",
)

dataloader_train_cosmos_h_surgical_json = L(get_generic_dataloader)(
    dataset=video_dataset_cosmos_h_surgical_json,
    sampler=L(get_sampler)(dataset=video_dataset_cosmos_h_surgical_json),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

_lora_defaults = [
    f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
    {"override /data_train": "mock"},
    {"override /data_val": "mock"},
    "_self_",
]

_lora_optimizer = dict(
    lr=1e-4,
    weight_decay=0.001,
)

_lora_scheduler = dict(
    f_max=[1.0],
    f_min=[0.2],
    warm_up_steps=[1000],
    cycle_lengths=[100000000],
    verbosity_interval=5,
)

_lora_callbacks = dict(
    heart_beat=dict(
        save_s3=False,
    ),
    iter_speed=dict(
        hit_thres=200,
        save_s3=False,
    ),
    device_monitor=dict(
        save_s3=False,
    ),
    every_n_sample_reg=dict(
        every_n=50000,
        save_s3=False,
    ),
    every_n_sample_ema=dict(
        every_n=50000,
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
)

_lora_model_config_first_frame_only = dict(
    config=dict(
        # Enable LoRA training
        use_lora=True,
        # LoRA configuration parameters
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights=True,
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
)

_lora_model_parallel = dict(
    context_parallel_size=1,
)

_lora_checkpoint_base = dict(
    load_from_object_store=dict(
        enabled=False,
    ),
    save_to_object_store=dict(
        enabled=False,
    ),
)

cosmos_h_surgical_predict_image2world_2b_lora = dict(
    defaults=_lora_defaults,
    job=dict(
        project="cosmos_h_surgical_predict",
        group="lora",
        name="cosmos_h_surgical_predict_image2world_2b_lora",
    ),
    dataloader_train=dataloader_train_cosmos_h_surgical_json,
    checkpoint=dict(
        **_lora_checkpoint_base,
        save_iter=50,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
    ),
    optimizer=_lora_optimizer,
    scheduler=_lora_scheduler,
    trainer=dict(
        logging_iter=5,
        max_iter=100_000,
        callbacks=_lora_callbacks,
    ),
    model=_lora_model_config_first_frame_only,
    model_parallel=_lora_model_parallel,
)


cs = ConfigStore.instance()

# Register the configurations with Hydra ConfigStore
for _item in [
    cosmos_h_surgical_predict_image2world_2b_lora,
]:
    # Get the experiment name from the global variable
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    # print("Registering experiment:", experiment_name)

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
