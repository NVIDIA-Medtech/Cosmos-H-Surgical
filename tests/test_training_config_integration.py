# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREDICT_RECIPE = ROOT / "examples" / "post_training" / "cosmos_h_surgical_predict_lora_480p.toml"
TRANSFER_RECIPE = ROOT / "examples" / "post_training" / "cosmos_h_surgical_transfer_lora_480p.toml"
DISTILL_RECIPE = ROOT / "examples" / "post_training" / "cosmos_h_surgical_transfer_dmd2_lora_480p_4step.toml"


@pytest.mark.parametrize(
    ("recipe", "prefix", "expected_name", "dataset_name"),
    [
        (PREDICT_RECIPE, "PREDICT", "cosmos_h_surgical_predict_lora_480p", "surgical_predict"),
        (TRANSFER_RECIPE, "TRANSFER", "cosmos_h_surgical_transfer_lora_480p", "surgical_transfer"),
    ],
)
def test_surgical_recipes_compose_with_pinned_framework(
    monkeypatch: pytest.MonkeyPatch,
    recipe: Path,
    prefix: str,
    expected_name: str,
    dataset_name: str,
) -> None:
    pytest.importorskip("webdataset")
    pytest.importorskip("multistorageclient")

    monkeypatch.setenv("COSMOS_TRAINING", "1")
    monkeypatch.setenv(f"COSMOS_H_SURGICAL_{prefix}_DATASET_DIRS", "/data/surgical")
    monkeypatch.setenv(f"COSMOS_H_SURGICAL_{prefix}_JSON_PATHS", "/data/surgical/manifests/train.json")
    monkeypatch.setenv(f"COSMOS_H_SURGICAL_{prefix}_ENLARGED_FACTORS", "1.0")
    monkeypatch.setenv("BASE_CHECKPOINT_PATH", "/models/Cosmos-H-Surgical-dcp")
    monkeypatch.setenv("WAN_VAE_PATH", "/models/Wan2.2_VAE.pth")

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.lazy_config.instantiate import instantiate

    from cosmos_h_surgical.training import register_experiments

    register_experiments()
    config = load_experiment_from_toml(recipe, extra_overrides=["trainer.max_iter=1"])

    assert config.job.name == expected_name
    assert config.trainer.max_iter == 1
    assert config.model.config.resolution == "480"
    assert config.model.config.lora_enabled is True
    assert config.model.config.action_gen is False
    assert config.optimizer.keys_to_select == ["lora_"]

    vlm_json = Path(config.model.config.vlm_config.model_instance.config.base_config.json_file)
    assert vlm_json.is_absolute()
    assert vlm_json.is_file()
    assert json.loads(vlm_json.read_text())["model_type"] == "qwen3_vl"
    assert (
        instantiate(config.model.config.vlm_config.model_instance.config.base_config).config_dict["model_type"]
        == "qwen3_vl"
    )

    assert config.checkpoint.load_path == "/models/Cosmos-H-Surgical-dcp"
    assert config.checkpoint.keys_to_skip_loading == ["net_ema."]
    assert config.dataloader_train.dataset_name == dataset_name
    assert config.dataloader_train.max_sequence_length == 45056
    dataset = config.dataloader_train.dataloader.datasets[dataset_name].dataset
    assert dataset.dataset_dir == "/data/surgical"
    assert dataset.json_path == "/data/surgical/manifests/train.json"
    assert dataset.enlarged_factor == "1.0"
    if prefix == "PREDICT":
        assert dataset.conditioning_config == {0: 0.2, 1: 0.7, 2: 0.1}
    else:
        assert dataset.control_modalities == {"edge": 2.0, "blur": 2.0, "depth": 2.0, "seg": 2.0}


def test_transfer_dmd2_recipe_composes_with_release_lora_teacher(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("webdataset")
    pytest.importorskip("multistorageclient")

    monkeypatch.setenv("COSMOS_TRAINING", "1")
    monkeypatch.setenv("COSMOS_H_SURGICAL_TRANSFER_DATASET_DIRS", "/data/surgical")
    monkeypatch.setenv("COSMOS_H_SURGICAL_TRANSFER_JSON_PATHS", "/data/surgical/manifests/train.json")
    monkeypatch.setenv("COSMOS_H_SURGICAL_TRANSFER_ENLARGED_FACTORS", "1.0")
    monkeypatch.setenv("COSMOS_H_SURGICAL_DISTILL_TEACHER_DCP", "/models/Cosmos-H-Surgical-release.dcp")
    monkeypatch.setenv("WAN_VAE_PATH", "/models/Wan2.2_VAE.pth")

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.trainer.distillation import DistillationTrainer

    from cosmos_h_surgical.distillation import SurgicalDMD2RFModel
    from cosmos_h_surgical.distillation_checkpointer import SurgicalDistillationCheckpointer
    from cosmos_h_surgical.dmd2_logging import DMD2WandBCallback
    from cosmos_h_surgical.training import register_experiments

    register_experiments()
    config = load_experiment_from_toml(DISTILL_RECIPE, extra_overrides=["trainer.max_iter=1"])

    assert config.job.name == "cosmos_h_surgical_transfer_dmd2_lora_480p_4step"
    assert config.trainer.type is DistillationTrainer
    assert config.trainer.max_iter == 1
    assert "iter_speed" in config.trainer.callbacks
    assert config.trainer.callbacks.iter_speed.every_n == 1
    assert config.trainer.callbacks.iter_speed.hit_thres == 0
    assert "dmd2_metrics" in config.trainer.callbacks
    assert config.trainer.callbacks.dmd2_metrics.output_path.endswith("/dmd2_metrics.jsonl")
    assert config.trainer.callbacks.dmd2_wandb._target_ is DMD2WandBCallback
    assert "termination_signal_checkpoint" in config.trainer.callbacks
    assert "wandb" not in config.trainer.callbacks
    assert config.model._target_ is SurgicalDMD2RFModel
    assert config.model.config.action_gen is False
    assert config.model.config.sound_gen is False
    assert config.model.config.lora_enabled is True
    assert config.model.config.vlm_config_teacher == config.model.config.vlm_config
    assert config.model.config.vlm_config_fake_score == config.model.config.vlm_config
    assert config.model.config.fixed_step_sampler_config.t_list == [1.0, 0.9375, 0.8333333333333334, 0.625]
    assert config.model.config.fixed_step_sampler_config.sample_type == "sde"
    assert config.model.config.simulation_mode == "backward"
    assert config.model.config.teacher_load_from.load_path == "/models/Cosmos-H-Surgical-release.dcp/model"
    assert config.optimizer.net.keys_to_select == ["lora_"]
    assert config.optimizer.fake_score.keys_to_select == ["lora_"]
    assert config.optimizer.net.lr == pytest.approx(5.0e-5)
    assert config.optimizer.fake_score.lr == pytest.approx(5.0e-5)
    assert config.checkpoint.type._target_ is SurgicalDistillationCheckpointer
    assert config.checkpoint.load_path == ""
    assert config.checkpoint.strict_resume is True
    assert config.dataloader_train.dataset_name == "surgical_transfer"
    dataset = config.dataloader_train.dataloader.datasets["surgical_transfer"].dataset
    assert dataset.control_modalities == {
        "edge": 1.0,
        "blur": 1.0,
        "depth": 1.0,
        "seg": 1.0,
    }
    assert dataset.seg_suffix == ".seg.mp4"
