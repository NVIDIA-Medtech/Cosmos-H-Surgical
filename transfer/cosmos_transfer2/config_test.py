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

from cosmos_transfer2.config import (
    MODEL_CHECKPOINTS,
    MODEL_KEYS,
    MULTICONTROL_OPENMDW_MODEL,
    CheckpointEdition,
    ModelKey,
    ModelVariant,
    SetupArguments,
)


def test_model_key():
    assert ModelKey().name == "edge"
    assert ModelKey(variant=ModelVariant.DEPTH).name == "depth"
    assert ModelKey(variant=ModelVariant.SEG, distilled=True).name == "seg/distilled"
    assert (
        ModelKey(
            variant=ModelVariant.VIS,
            checkpoint_edition=CheckpointEdition.OPENMDW_1_1,
        ).name
        == "vis/openmdw-1.1"
    )


def test_openmdw_checkpoints():
    expected = {
        ModelVariant.DEPTH: (
            "88ee68f4-6e02-4ff8-86a6-8a5b16df85d0",
            "openmdw-1.1/transfer/depth/cosmos-h-surgical-transfer-depth_model_ema_bf16.pt",
        ),
        ModelVariant.EDGE: (
            "291746d0-4851-44a7-aa29-71b4a069b167",
            "openmdw-1.1/transfer/edge/cosmos-h-surgical-transfer-edge_model_ema_bf16.pt",
        ),
        ModelVariant.SEG: (
            "2db84acf-a75f-4a7f-b357-cfdebb0b69f4",
            "openmdw-1.1/transfer/seg/cosmos-h-surgical-transfer-seg_model_ema_bf16.pt",
        ),
        ModelVariant.VIS: (
            "3f518e70-fdb2-44e6-96cd-95aa16dc259c",
            "openmdw-1.1/transfer/vis/cosmos-h-surgical-transfer-vis_model_ema_bf16.pt",
        ),
    }

    for variant, (uuid, filename) in expected.items():
        key = ModelKey(variant=variant, checkpoint_edition=CheckpointEdition.OPENMDW_1_1)
        checkpoint = MODEL_CHECKPOINTS[key]
        assert checkpoint.uuid == uuid
        assert checkpoint.hf.revision == "92d2558c3329c91fa77002f6604987ba8a6ce29a"
        assert checkpoint.hf.filename == filename

        args = SetupArguments(output_dir=Path("/tmp/output"), model=key.name)
        assert args.model_key == key
        assert args.checkpoint_path == checkpoint.s3.uri


def test_multicontrol_openmdw_alias():
    key = MODEL_KEYS[MULTICONTROL_OPENMDW_MODEL]
    assert key.checkpoint_edition == CheckpointEdition.OPENMDW_1_1

    args = SetupArguments(output_dir=Path("/tmp/output"), model=MULTICONTROL_OPENMDW_MODEL)
    assert args.model == "multicontrol/openmdw-1.1"
    assert args.model_key == key
    assert args.checkpoint_path == MODEL_CHECKPOINTS[key].s3.uri
