# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Preflight validation for the released Surgical DMD2 teacher artifact."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

LORA_TARGET_MODULES = (
    "q_proj_moe_gen",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
    "o_proj_moe_gen",
)
EXPECTED_LORA_RANK = 16
EXPECTED_LORA_ALPHA = 32
EXPECTED_LORA_TENSORS = 288
EXPECTED_ACTION_TENSORS = {
    "model.net.action2llm.bias.weight",
    "model.net.action2llm.fc.weight",
    "model.net.action_modality_embed",
    "model.net.llm2action.bias.weight",
    "model.net.llm2action.fc.weight",
}


@dataclass(frozen=True)
class TeacherCheckpointReport:
    checkpoint_path: str
    tensors: int
    lora_tensors: int
    action_tensors: int
    shards: int
    shard_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read valid JSON from {path}: {error}") from error


def validate_release_teacher_checkpoint(checkpoint_path: Path) -> TeacherCheckpointReport:
    """Validate Nano lineage, released LoRA metadata, index, and shards."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise ValueError(f"Teacher checkpoint directory does not exist: {checkpoint_path}")

    config_path = checkpoint_path / "config.json"
    index_path = checkpoint_path / "model.safetensors.index.json"
    config = _read_json(config_path)
    index = _read_json(index_path)
    try:
        model_config = config["model"]["config"]
        weight_map = index["weight_map"]
    except (KeyError, TypeError) as error:
        raise ValueError("Teacher checkpoint is missing the Cosmos3 model config or safetensors weight map.") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Teacher safetensors weight map is empty.")

    expected_targets = ",".join(LORA_TARGET_MODULES)
    checks = {
        "Cosmos3 Nano model": model_config.get("vlm_config", {}).get("model_name") == "Qwen/Qwen3-VL-8B-Instruct",
        "vision generation": model_config.get("vision_gen") is True,
        "released action capability": model_config.get("action_gen") is True,
        "sound disabled": model_config.get("sound_gen") is False,
        "480P resolution": str(model_config.get("resolution")) == "480",
        "LoRA enabled": model_config.get("lora_enabled") is True,
        "LoRA rank": model_config.get("lora_rank") == EXPECTED_LORA_RANK,
        "LoRA alpha": model_config.get("lora_alpha") == EXPECTED_LORA_ALPHA,
        "LoRA targets": model_config.get("lora_target_modules") == expected_targets,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Teacher checkpoint failed release-config checks: {', '.join(failed)}")

    keys = set(weight_map)
    lora_keys = sorted(key for key in keys if ".lora_" in key)
    if len(lora_keys) != EXPECTED_LORA_TENSORS:
        raise ValueError(f"Expected {EXPECTED_LORA_TENSORS} released LoRA tensors, found {len(lora_keys)}.")
    for module_name in LORA_TARGET_MODULES:
        module_lora_keys = [key for key in lora_keys if f".{module_name}.lora_" in key]
        if len(module_lora_keys) != EXPECTED_LORA_TENSORS // len(LORA_TARGET_MODULES):
            raise ValueError(
                f"Expected {EXPECTED_LORA_TENSORS // len(LORA_TARGET_MODULES)} LoRA tensors for "
                f"{module_name}, found {len(module_lora_keys)}."
            )
    invalid_lora_keys = [key for key in lora_keys if not key.endswith((".lora_A.weight", ".lora_B.weight"))]
    if invalid_lora_keys:
        raise ValueError(f"Unexpected released LoRA tensor names: {invalid_lora_keys[:20]}")

    action_keys = {key for key in keys if "action" in key.lower()}
    if action_keys != EXPECTED_ACTION_TENSORS:
        raise ValueError(
            "Released action-only tensor inventory changed: "
            f"missing={sorted(EXPECTED_ACTION_TENSORS - action_keys)}, "
            f"unexpected={sorted(action_keys - EXPECTED_ACTION_TENSORS)}"
        )

    shard_names = sorted(set(weight_map.values()))
    shard_bytes = 0
    for shard_name in shard_names:
        shard_path = checkpoint_path / shard_name
        if not shard_path.is_file():
            raise ValueError(f"Teacher checkpoint shard is missing: {shard_path}")
        size = shard_path.stat().st_size
        if size <= 0:
            raise ValueError(f"Teacher checkpoint shard is empty: {shard_path}")
        shard_bytes += size

    return TeacherCheckpointReport(
        checkpoint_path=str(checkpoint_path),
        tensors=len(keys),
        lora_tensors=len(lora_keys),
        action_tensors=len(action_keys),
        shards=len(shard_names),
        shard_bytes=shard_bytes,
    )
