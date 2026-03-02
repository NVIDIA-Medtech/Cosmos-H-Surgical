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

"""Generic video dataset loader for Cosmos Predict2."""

import json
import math
import os
import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state
from torch.utils.data import Dataset, DistributedSampler
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo

DATA_LOADER_SEED = int(os.environ.get("DATA_LOADER_SEED", 0))


class VideoDatasetJSON(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        json_path: str,
        enlarged_factor: str,
        num_frames: int,
        video_size: tuple[int, int],
        caption_format: str = "text",  # "text", "json", or "auto"
        prompt_type: str = "short",  # "long", "short", "medium", or None for auto
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Dataset structure:
        dataset_dir/
            ├── video1.mp4
            └── video2.mp4
            ├── video1.json  ({"short": "text description"})
            └── video2.json

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.caption_format = caption_format
        self.prompt_type = prompt_type
        self.dataset_dir = dataset_dir.split(",")
        self.json_path = json_path.split(",")
        self.enlarged_factor = [float(f) for f in enlarged_factor.split(",")]
        assert len(self.dataset_dir) == len(self.json_path) == len(self.enlarged_factor), (
            "dataset_dir, json_path and enlarged_factor must have the same length"
        )
        print(f"Loading data from {len(self.dataset_dir)} datasets")
        self.sequence_length = num_frames

        self.video_paths = []
        for dd, jp, ef in zip(self.dataset_dir, self.json_path, self.enlarged_factor):
            with open(jp, "r") as data:
                data_list = json.load(data)
            if "training" in data_list:
                data_list = data_list["training"]
                print(f"Loading data from training split")
            else:
                raise NotImplementedError

            data_list = sorted([os.path.join(dd, data_dict["video"]) for data_dict in data_list])
            original_length = len(data_list)
            random.seed(DATA_LOADER_SEED)
            random.shuffle(data_list)
            print(
                f"json_path: {jp}, rank {parallel_state.get_data_parallel_rank()}: Shuffled data list: {data_list[:5]} ... {data_list[-5:]}"
            )

            if ef >= 1:
                data_list = (data_list * math.ceil(ef))[: int(original_length * ef)]
            else:
                data_list = data_list[: int(len(data_list) * ef)]

            log.warning(
                f"json_path: {jp} enlarged factor is {ef}, original length is {original_length}, new length is {len(data_list)}"
            )

            self.video_paths += data_list

        self.video_paths = sorted(self.video_paths)
        log.warning(f"{len(self.video_paths)} videos in total")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )
        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        if max_start_idx <= 0:
            start_frame = 0
        else:
            start_frame = np.random.randint(0, max_start_idx)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            log.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    captions = json.loads("{" + content + "}")
                else:
                    captions = json.loads(content)

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    log.warning(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            log.warning(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video, fps = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
            video_path = self.video_paths[index]

            if self.caption_format == "json":
                caption_path = video_path.replace(".mp4", ".json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = video_path.replace(".mp4", ".txt")
                caption = self._load_text(Path(caption_path))

            data["video"] = video
            data["ai_caption"] = caption
            log.debug(f"Loading video: {video_path}, index: {index}, caption: {caption}")

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    print(f"Using data loader seed: {DATA_LOADER_SEED}")
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=DATA_LOADER_SEED,
    )


if __name__ == "__main__":
    # test the dataset
    dataset = VideoDatasetJSON(
        dataset_dir="/dataset_dir1,/dataset_dir2,/dataset_dir3",
        json_path="/json_path1,/json_path2,/json_path3",
        enlarged_factor="2.1,0.5,0.3",
        num_frames=93,
        video_size=(704, 1280),
        caption_format="json",
    )
    print(dataset)
    for i in range(20):
        sample = dataset[i]
        print(f"Sample {i}: video shape {sample['video'].shape}, caption: {sample['ai_caption']}, fps: {sample['fps']}")
