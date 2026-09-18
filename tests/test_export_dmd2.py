# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

from cosmos_h_surgical.checkpoints import BASE_MODEL_CONFIG_PATH
from cosmos_h_surgical.export_dmd2 import sanitize_student_export_config


def test_sanitize_student_export_config_replaces_run_local_paths(tmp_path: Path) -> None:
    config = json.loads(BASE_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = config["model"]["config"]
    model_config["fixed_step_sampler_config"] = {
        "sample_type": "sde",
        "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
    }
    model_config["tokenizer"]["bucket_name"] = "bucket"
    model_config["tokenizer"]["vae_path"] = "/workspace/cache/Wan2.2_VAE.pth"
    model_config["tokenizer"]["object_store_credential_path_pretrained"] = "/home/user/credentials"
    model_config["vlm_config"]["pretrained_weights"]["backbone_path"] = "/workspace/cache/reasoner"
    model_config["vlm_config"]["pretrained_weights"]["credentials_path"] = "/home/user/credentials"
    model_config["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"] = (
        "/workspace/framework/Qwen3-VL-8B-Instruct.json"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    sanitize_student_export_config(tmp_path)

    sanitized_text = config_path.read_text(encoding="utf-8")
    sanitized = json.loads(sanitized_text)["model"]["config"]
    public = json.loads(BASE_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))["model"]["config"]
    assert sanitized["tokenizer"]["bucket_name"] == "bucket"
    assert sanitized["tokenizer"]["vae_path"] == public["tokenizer"]["vae_path"]
    assert sanitized["tokenizer"]["object_store_credential_path_pretrained"] == ""
    assert sanitized["vlm_config"]["pretrained_weights"]["backbone_path"] == ""
    assert sanitized["vlm_config"]["pretrained_weights"]["credentials_path"] == ""
    assert (
        sanitized["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"]
        == public["vlm_config"]["model_instance"]["config"]["base_config"]["json_file"]
    )
    assert "/workspace/" not in sanitized_text
    assert "/home/" not in sanitized_text
