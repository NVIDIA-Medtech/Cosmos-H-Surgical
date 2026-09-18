# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict DMD2 checkpoint resume with unstepped-optimizer compatibility."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from cosmos_framework.checkpoint.dcp import CustomLoadPlanner, _DataloaderWrapper
from cosmos_framework.checkpoint.dcp_distill import (
    DistributedCheckpointer as FrameworkDistillationCheckpointer,
)
from cosmos_framework.checkpoint.dcp_distill import ModelWrapper, OptimizerWrapper
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log, misc
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict

_FUSED_ADAM_MOMENT_SUFFIXES = (".exp_avg", ".exp_avg_sq")


def reconcile_optimizer_state_for_strict_load(
    state_dict: Mapping[str, Any],
    checkpoint_keys: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Match a restore skeleton to a valid partially initialized optimizer.

    FusedAdam initializes moments lazily. A checkpoint written before one DMD2
    phase has stepped therefore contains complete parameter groups but no Adam
    moments for that phase. A fresh restore skeleton can already expose zero
    moment leaves, which the default strict DCP planner mistakes for missing
    checkpoint data. Only those lazy moment leaves may be omitted.
    """
    state_keys = set(state_dict)
    unexpected = sorted(checkpoint_keys - state_keys)
    omitted = sorted(state_keys - checkpoint_keys)
    invalid_omissions = [
        key for key in omitted if not (key.startswith("state.") and key.endswith(_FUSED_ADAM_MOMENT_SUFFIXES))
    ]
    if unexpected or invalid_omissions:
        raise ValueError(
            "Strict optimizer resume schema mismatch: "
            f"unexpected_checkpoint_keys={unexpected[:20]}, "
            f"missing_non_moment_keys={invalid_omissions[:20]}"
        )
    return dict(state_dict), omitted


class SurgicalDistillationCheckpointer(FrameworkDistillationCheckpointer):
    """Framework DMD2 checkpointer with strict lazy-FusedAdam resume support."""

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: Any = None,
        scheduler: Any = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        model_dict = model.model_dict()
        resume_keys, checkpoint_path, warm_start = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")
        iteration = 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            for key in resume_keys:
                dist.barrier()
                component_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {component_path}")

                strict_resume = self.config_checkpoint.strict_resume
                keys_to_skip_loading = self.config_checkpoint.keys_to_skip_loading if warm_start else []
                load_planner = CustomLoadPlanner(
                    allow_partial_load=not strict_resume,
                    keys_to_skip_loading=keys_to_skip_loading,
                )

                if key == "model":
                    storage_reader = self.get_storage_reader(component_path)
                    log.info("- Loading the model...")
                    model_wrapper = ModelWrapper(
                        model,
                        exclude_teacher_weights=True,
                        strict_resume=strict_resume,
                    )
                    state_dict = model_wrapper.state_dict()
                    dcp.load(state_dict, storage_reader=storage_reader, planner=load_planner)
                    model_wrapper.load_state_dict(state_dict)

                elif key == "optim":
                    if optimizer is None:
                        raise ValueError("optimizer must be provided when loading optim state.")
                    for optimizer_key, phase_optimizer in optimizer.items():
                        storage_reader = self.get_storage_reader(f"{component_path}_{optimizer_key}")
                        log.info(f"- Loading the optimizer ({optimizer_key})...")
                        optimizer_wrapper = OptimizerWrapper(model_dict[optimizer_key], phase_optimizer)
                        state_dict = optimizer_wrapper.state_dict()
                        if strict_resume:
                            checkpoint_keys = set(storage_reader.read_metadata().state_dict_metadata)
                            state_dict, omitted = reconcile_optimizer_state_for_strict_load(
                                state_dict,
                                checkpoint_keys,
                            )
                            if omitted:
                                log.warning(
                                    f"Optimizer {optimizer_key} has {len(omitted)} uninitialized Adam moment "
                                    "leaf/leaves in the restore skeleton; preserving the checkpoint's "
                                    "unstepped state."
                                )
                        optimizer_load_planner = CustomLoadPlanner(
                            allow_partial_load=not strict_resume or bool(omitted),
                            keys_to_skip_loading=[],
                        )
                        dcp.load(
                            state_dict,
                            storage_reader=storage_reader,
                            planner=optimizer_load_planner,
                        )
                        optimizer_wrapper.load_state_dict(state_dict)

                elif key == "scheduler":
                    if scheduler is None:
                        raise ValueError("scheduler must be provided when loading scheduler state.")
                    for scheduler_key, phase_scheduler in scheduler.items():
                        storage_reader = self.get_storage_reader(f"{component_path}_{scheduler_key}")
                        log.info(f"- Loading the scheduler ({scheduler_key})...")
                        state_dict = phase_scheduler.state_dict()
                        dcp.load(state_dict, storage_reader=storage_reader, planner=load_planner)
                        phase_scheduler.load_state_dict(state_dict)

                elif key == "trainer":
                    if grad_scaler is None:
                        raise ValueError("grad_scaler must be provided when loading trainer state.")
                    storage_reader = self.get_storage_reader(component_path)
                    log.info("- Loading the trainer...")

                    rng_key = f"rng_state_{dist.get_rank()}"
                    current_rng_state = get_rand_state_dict()
                    state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    metadata = storage_reader.read_metadata()
                    if any(item.startswith(f"{rng_key}.") or item == rng_key for item in metadata.state_dict_metadata):
                        state_dict[rng_key] = current_rng_state
                    dcp.load(state_dict, storage_reader=storage_reader, planner=load_planner)
                    grad_scaler.load_state_dict(state_dict["grad_scaler"])
                    iteration = state_dict["iteration"]
                    set_rand_state_dict(state_dict.get(rng_key, current_rng_state))

                elif key == "dataloader":
                    if not easy_io.exists(component_path, backend_key=self.load_s3_backend_key):
                        log.info(
                            f"Checkpoint {component_path} does not exist, skip loading dataloader.",
                            rank0_only=False,
                        )
                        continue
                    rank = dist.get_rank()
                    dataloader_path = os.path.join(component_path, f"rank_{rank}.pkl")
                    if not easy_io.exists(dataloader_path, backend_key=self.load_s3_backend_key):
                        log.info(f"No dataloader checkpoint found at {dataloader_path}", rank0_only=False)
                        continue
                    log.info(f"- Loading the dataloader {component_path}...", rank0_only=False)
                    state_dict = easy_io.load(
                        dataloader_path,
                        file_format="pkl",
                        backend_key=self.load_s3_backend_key,
                    )
                    dataloader_wrapper = _DataloaderWrapper(self.callbacks)
                    if dataloader_wrapper.has_state():
                        dataloader_wrapper.load_state_dict(state_dict)
                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")

            if self.callbacks is not None:
                self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            log.info(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")
        else:
            log.info("Training from scratch.")

        torch.cuda.empty_cache()
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration
