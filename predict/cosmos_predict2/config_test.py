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

from pathlib import Path

from cosmos_predict2.config import (
    MODEL_CHECKPOINTS,
    CheckpointEdition,
    ModelKey,
    ModelSize,
    ModelVariant,
    SetupArguments,
)


def test_model_key():
    assert ModelKey().name == "2B/post-trained"
    assert ModelKey(size=ModelSize._14B).name == "14B/post-trained"
    assert ModelKey(variant=ModelVariant.AUTO_MULTIVIEW).name == "2B/auto/multiview"
    assert ModelKey(checkpoint_edition=CheckpointEdition.OPENMDW_1_1).name == "2B/post-trained/openmdw-1.1"


def test_openmdw_checkpoint():
    key = ModelKey(checkpoint_edition=CheckpointEdition.OPENMDW_1_1)
    checkpoint = MODEL_CHECKPOINTS[key]

    assert checkpoint.uuid == "930af493-8e65-4dbb-b6d5-4b61226e9a44"
    assert checkpoint.hf.revision == "92d2558c3329c91fa77002f6604987ba8a6ce29a"
    assert checkpoint.hf.filename == ("openmdw-1.1/predict/cosmos-h-surgical-predict_model_ema_bf16.pt")

    args = SetupArguments(output_dir=Path("/tmp/output"), model=key.name)
    assert args.model_key == key
    assert args.checkpoint_path == checkpoint.s3.uri
