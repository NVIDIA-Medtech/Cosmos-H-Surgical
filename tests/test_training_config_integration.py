# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "examples" / "post_training" / "cosmos_h_surgical_vision_lora_480p.toml"


def test_surgical_recipe_composes_with_pinned_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("webdataset")
    pytest.importorskip("multistorageclient")

    monkeypatch.setenv("COSMOS_TRAINING", "1")
    monkeypatch.setenv("COSMOS_H_SURGICAL_DATASET", "/tmp/surgical_train.jsonl")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/tmp/cosmos3-base")
    monkeypatch.setenv("WAN_VAE_PATH", "/tmp/Wan2.2_VAE.pth")

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    from cosmos_h_surgical.training import register_experiments

    register_experiments()
    config = load_experiment_from_toml(RECIPE, extra_overrides=["trainer.max_iter=1"])

    assert config.job.name == "cosmos_h_surgical_vision_lora_480p"
    assert config.trainer.max_iter == 1
    assert config.model.config.resolution == "480"
    assert config.model.config.lora_enabled is True
    assert config.model.config.action_gen is False
    assert config.optimizer.keys_to_select == ["lora_"]
    assert config.dataloader_train.max_sequence_length == 45056
