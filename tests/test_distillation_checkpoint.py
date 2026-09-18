# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest

from cosmos_h_surgical.distillation_checkpoint import (
    EXPECTED_LORA_TENSORS,
    LORA_TARGET_MODULES,
    validate_release_teacher_checkpoint,
)


def _write_release_checkpoint(root: Path) -> None:
    model_config = {
        "vision_gen": True,
        "action_gen": True,
        "sound_gen": False,
        "resolution": "480",
        "lora_enabled": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_target_modules": ",".join(LORA_TARGET_MODULES),
        "vlm_config": {"model_name": "Qwen/Qwen3-VL-8B-Instruct"},
    }
    (root / "config.json").write_text(json.dumps({"model": {"config": model_config}}), encoding="utf-8")

    keys = {
        f"model.net.language_model.model.layers.{layer}.self_attn.{module}.lora_{adapter}.weight"
        for layer in range(36)
        for module in LORA_TARGET_MODULES
        for adapter in ("A", "B")
    }
    keys.update(
        {
            "model.net.action2llm.bias.weight",
            "model.net.action2llm.fc.weight",
            "model.net.action_modality_embed",
            "model.net.llm2action.bias.weight",
            "model.net.llm2action.fc.weight",
            "model.net.language_model.model.embed_tokens.weight",
        }
    )
    assert len([key for key in keys if ".lora_" in key]) == EXPECTED_LORA_TENSORS
    shard_name = "model-00001-of-00001.safetensors"
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard_name for key in sorted(keys)}}),
        encoding="utf-8",
    )
    (root / shard_name).write_bytes(b"checkpoint")


def test_validate_release_teacher_checkpoint_accepts_release_lora(tmp_path: Path) -> None:
    _write_release_checkpoint(tmp_path)
    report = validate_release_teacher_checkpoint(tmp_path)
    assert report.lora_tensors == 288
    assert report.action_tensors == 5
    assert report.shards == 1
    assert report.shard_bytes == len(b"checkpoint")


def test_validate_release_teacher_checkpoint_rejects_missing_lora(tmp_path: Path) -> None:
    _write_release_checkpoint(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    lora_key = next(key for key in index["weight_map"] if ".lora_" in key)
    del index["weight_map"][lora_key]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 288 released LoRA tensors"):
        validate_release_teacher_checkpoint(tmp_path)


def test_validate_release_teacher_checkpoint_rejects_empty_shard(tmp_path: Path) -> None:
    _write_release_checkpoint(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="shard is empty"):
        validate_release_teacher_checkpoint(tmp_path)
