# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_h_surgical.distillation_checkpointer import reconcile_optimizer_state_for_strict_load


def test_strict_optimizer_reconcile_allows_uninitialized_fused_adam_moments() -> None:
    state_dict = {
        "param_groups.net.layer.lora_A.weight.lr": 5.0e-5,
        "param_groups.net.layer.lora_A.weight.step": 0,
        "state.net.layer.lora_A.weight.exp_avg": object(),
        "state.net.layer.lora_A.weight.exp_avg_sq": object(),
    }
    checkpoint_keys = {
        "param_groups.net.layer.lora_A.weight.lr",
        "param_groups.net.layer.lora_A.weight.step",
    }

    reconciled, omitted = reconcile_optimizer_state_for_strict_load(state_dict, checkpoint_keys)

    assert set(reconciled) == set(state_dict)
    assert omitted == [
        "state.net.layer.lora_A.weight.exp_avg",
        "state.net.layer.lora_A.weight.exp_avg_sq",
    ]


def test_strict_optimizer_reconcile_rejects_missing_parameter_group_state() -> None:
    state_dict = {
        "param_groups.net.layer.lora_A.weight.lr": 5.0e-5,
        "param_groups.net.layer.lora_A.weight.step": 0,
    }

    with pytest.raises(ValueError, match="missing_non_moment_keys"):
        reconcile_optimizer_state_for_strict_load(
            state_dict,
            {"param_groups.net.layer.lora_A.weight.lr"},
        )


def test_strict_optimizer_reconcile_rejects_unexpected_checkpoint_state() -> None:
    state_dict = {"param_groups.net.layer.lora_A.weight.lr": 5.0e-5}

    with pytest.raises(ValueError, match="unexpected_checkpoint_keys"):
        reconcile_optimizer_state_for_strict_load(
            state_dict,
            {
                "param_groups.net.layer.lora_A.weight.lr",
                "state.net.unknown.weight.exp_avg",
            },
        )
