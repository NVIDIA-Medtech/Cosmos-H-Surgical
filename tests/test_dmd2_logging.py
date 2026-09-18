# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from cosmos_h_surgical.dmd2_logging import DMD2WandBCallback


class _Timer:
    def __init__(self) -> None:
        self.reset_calls = 0

    def compute_average_results(self) -> dict[str, float]:
        return {"forward": 1.25}

    def reset(self) -> None:
        self.reset_calls += 1


def test_dmd2_wandb_callback_initializes_and_logs_phase_metrics(monkeypatch) -> None:
    callback = DMD2WandBCallback()
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=1))
    timer = _Timer()
    callback.trainer = SimpleNamespace(sample_counter=64, training_timer=timer)
    model = SimpleNamespace(
        get_phase=lambda iteration: "student" if iteration % 5 == 0 else "critic",
        _distillation_parity_grad_metrics={"dmd_vsd_grad_norm": torch.tensor(0.75)},
    )
    optimizer = SimpleNamespace(
        items=lambda: iter(
            {
                "net": torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=5.0e-5, weight_decay=0.0),
                "fake_score": torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=4.0e-5, weight_decay=0.1),
            }.items()
        )
    )
    init_wandb = MagicMock()
    wandb_log = MagicMock()
    monkeypatch.setattr("cosmos_h_surgical.dmd2_logging.wandb_util.init_wandb", init_wandb)
    monkeypatch.setattr("cosmos_h_surgical.dmd2_logging.distributed.is_rank0", lambda: True)
    monkeypatch.setattr("cosmos_h_surgical.dmd2_logging.wandb.run", object())
    monkeypatch.setattr("cosmos_h_surgical.dmd2_logging.wandb.log", wandb_log)

    callback.on_train_start(model, iteration=0)
    callback.on_before_optimizer_step(model, optimizer, scheduler=object(), grad_scaler=object(), iteration=0)
    callback.on_training_step_end(
        model,
        data_batch={},
        output_batch={"vsd_loss": torch.tensor(2.0), "total_generator_loss": torch.tensor(3.0)},
        loss=torch.tensor(4.0),
        iteration=1,
    )

    init_wandb.assert_called_once_with(callback.config, model=model)
    info = wandb_log.call_args.args[0]
    assert wandb_log.call_args.kwargs == {"step": 1}
    assert info["train/loss"] == 4.0
    assert info["train/phase"] == "student"
    assert info["train/vsd_loss"] == 2.0
    assert info["train/total_generator_loss"] == 3.0
    assert info["train/dmd_vsd_grad_norm"] == 0.75
    assert info["optim/net_lr_0"] == 5.0e-5
    assert info["optim/fake_score_lr_0"] == 4.0e-5
    assert info["timer/forward"] == 1.25
    assert timer.reset_calls == 1


def test_dmd2_wandb_callback_honors_logging_interval(monkeypatch) -> None:
    callback = DMD2WandBCallback()
    callback.config = SimpleNamespace(trainer=SimpleNamespace(logging_iter=10))
    timer = _Timer()
    callback.trainer = SimpleNamespace(training_timer=timer)
    wandb_log = MagicMock()
    monkeypatch.setattr("cosmos_h_surgical.dmd2_logging.wandb.run", object())
    monkeypatch.setattr("cosmos_h_surgical.dmd2_logging.wandb.log", wandb_log)

    callback.on_training_step_end(
        SimpleNamespace(get_phase=lambda _iteration: "critic"),
        data_batch={},
        output_batch={"fake_score_loss": torch.tensor(2.0)},
        loss=torch.tensor(2.0),
        iteration=3,
    )

    wandb_log.assert_not_called()
    assert timer.reset_calls == 0
