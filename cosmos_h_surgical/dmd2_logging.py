# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""W&B logging compatible with alternating-phase DMD2 training."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
import wandb
from cosmos_framework.callbacks.dmd2_metrics import PARITY_KEYS
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, wandb_util
from cosmos_framework.utils.callback import Callback


def _phase(model: torch.nn.Module, iteration: int) -> str:
    get_phase = getattr(model, "get_phase", None)
    if callable(get_phase):
        return str(get_phase(iteration))
    get_optimizer_key = getattr(model, "get_optimizer_key", None)
    if callable(get_optimizer_key):
        return "student" if str(get_optimizer_key(iteration)) == "net" else "critic"
    raise AttributeError(f"{type(model).__name__} must define get_phase() or get_optimizer_key()")


def _global_mean(value: object, device: torch.device) -> float:
    if isinstance(value, torch.Tensor):
        to_local = getattr(value, "to_local", None)
        local_value = to_local() if callable(to_local) else value
        metric = local_value.detach().float().mean().to(device)
    else:
        metric = torch.tensor(float(value), device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(metric, op=dist.ReduceOp.AVG)
    return metric.item()


class DMD2WandBCallback(Callback):
    """Initialize W&B and log DMD2 metrics without assuming one scheduler."""

    def __init__(self) -> None:
        super().__init__()
        self._optimizer: object | None = None

    @staticmethod
    def _iter_named_optimizers(optimizer: object) -> Iterator[tuple[str, object]]:
        items = getattr(optimizer, "items", None)
        if callable(items):
            yield from items()
        else:
            yield "optimizer", optimizer

    @staticmethod
    def _iter_param_groups(optimizer: object) -> Iterator[dict[str, Any]]:
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups is not None:
            yield from param_groups
            return
        for inner_optimizer in getattr(optimizer, "optimizers", ()):
            yield from inner_optimizer.param_groups

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del iteration
        wandb_util.init_wandb(self.config, model=model)

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: object,
        scheduler: object,
        grad_scaler: object,
        iteration: int = 0,
    ) -> None:
        del model, scheduler, grad_scaler, iteration
        self._optimizer = optimizer

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del data_batch
        if iteration % int(self.config.trainer.logging_iter) != 0:
            return

        grad_metrics = getattr(model, "_distillation_parity_grad_metrics", {})
        metric_values = {**output_batch, **grad_metrics}
        info: dict[str, Any] = {
            "iteration": iteration,
            "sample_counter": getattr(self.trainer, "sample_counter", iteration),
            "train/loss": _global_mean(loss, loss.device),
            "train/phase": _phase(model, max(iteration - 1, 0)),
        }
        for key in PARITY_KEYS:
            if key in metric_values:
                info[f"train/{key}"] = _global_mean(metric_values[key], loss.device)

        timer_results = self.trainer.training_timer.compute_average_results()
        info.update({f"timer/{key}": value for key, value in timer_results.items()})
        if self._optimizer is not None:
            for optimizer_name, optimizer in self._iter_named_optimizers(self._optimizer):
                for group_index, param_group in enumerate(self._iter_param_groups(optimizer)):
                    info[f"optim/{optimizer_name}_lr_{group_index}"] = param_group["lr"]
                    if "weight_decay" in param_group:
                        info[f"optim/{optimizer_name}_weight_decay_{group_index}"] = param_group["weight_decay"]

        if distributed.is_rank0() and wandb.run is not None:
            wandb.log(info, step=iteration)
        self.trainer.training_timer.reset()

    @distributed.rank0_only
    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model, iteration
        if wandb.run is not None:
            wandb.finish()


__all__: tuple[str, ...] = ("DMD2WandBCallback",)
