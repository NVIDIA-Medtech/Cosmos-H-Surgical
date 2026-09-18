# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Four-step LoRA DMD2 distillation for surgical transfer."""

import copy
from collections.abc import Mapping

from cosmos_framework.callbacks.dmd2_metrics import DMD2Metrics
from cosmos_framework.callbacks.iter_speed import IterSpeed
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2OptimizerConfig, DMD2RFConfig
from cosmos_framework.trainer.distillation import DistillationTrainer
from cosmos_framework.utils.generator.optimizer import build_optimizer
from cosmos_framework.utils.lazy_config import LazyCall as L
from hydra.core.config_store import ConfigStore

from cosmos_h_surgical.configs.training_common import (
    common_dataset_kwargs,
    make_packing_dataloader,
    make_surgical_lora_config,
)
from cosmos_h_surgical.data.surgical_transfer_json_dataset import get_surgical_transfer_json_dataset
from cosmos_h_surgical.distillation import SurgicalDMD2RFModel
from cosmos_h_surgical.distillation_checkpointer import SurgicalDistillationCheckpointer
from cosmos_h_surgical.dmd2_logging import DMD2WandBCallback

EXPERIMENT_NAME = "cosmos_h_surgical_transfer_dmd2_lora_480p_4step"
FIXED_STEP_SCHEDULE = [1.0, 15 / 16, 5 / 6, 5 / 8]


def _dmd2_optimizer(lr: float) -> dict:
    return L(build_optimizer)(
        model=None,
        optimizer_type="FusedAdam",
        lr=lr,
        weight_decay=0.0,
        betas=[0.0, 0.999],
        fused=True,
        eps=1.0e-8,
        keys_to_select=["lora_"],
        lr_multipliers={},
    )


def _register_distillation_groups() -> None:
    config_store = ConfigStore.instance()
    config_store.store(
        group="model",
        package="_global_",
        name="surgical_dmd2_fsdp",
        node={
            "trainer": {
                "distributed_parallelism": "fsdp",
            },
            "model": L(SurgicalDMD2RFModel)(
                config=DMD2RFConfig(
                    parallelism=ParallelismConfig(data_parallel_shard_degree=8),
                ),
                _recursive_=False,
            ),
        },
    )
    config_store.store(
        group="optimizer",
        package="optimizer",
        name="surgical_dmd2_lora",
        node=DMD2OptimizerConfig(
            net=_dmd2_optimizer(5.0e-5),
            fake_score=_dmd2_optimizer(5.0e-5),
        ),
    )
    config_store.store(
        group="ckpt_type",
        package="checkpoint.type",
        name="surgical_dmd2_dcp",
        node=L(SurgicalDistillationCheckpointer)(),
    )


def _replace_default(config: object, group: str, choice: str) -> None:
    defaults = config.defaults
    for index, entry in enumerate(defaults):
        if isinstance(entry, Mapping) and f"override /{group}" in entry:
            defaults[index] = {f"override /{group}": choice}
            return
    raise ValueError(f"Unable to replace missing Hydra default group: {group}")


_register_distillation_groups()

cosmos_h_surgical_transfer_dmd2_lora_480p_4step = make_surgical_lora_config(
    experiment_name=EXPERIMENT_NAME,
    group="transfer_dmd2_lora_480p_4step",
)
_replace_default(cosmos_h_surgical_transfer_dmd2_lora_480p_4step, "model", "surgical_dmd2_fsdp")
_replace_default(cosmos_h_surgical_transfer_dmd2_lora_480p_4step, "optimizer", "surgical_dmd2_lora")
_replace_default(cosmos_h_surgical_transfer_dmd2_lora_480p_4step, "checkpoint", "local")
_replace_default(cosmos_h_surgical_transfer_dmd2_lora_480p_4step, "ckpt_type", "surgical_dmd2_dcp")
for index, entry in enumerate(cosmos_h_surgical_transfer_dmd2_lora_480p_4step.defaults):
    if isinstance(entry, Mapping) and "override /callbacks" in entry:
        cosmos_h_surgical_transfer_dmd2_lora_480p_4step.defaults[index] = {"override /callbacks": ["job_monitor"]}
        break

model_config = cosmos_h_surgical_transfer_dmd2_lora_480p_4step.model.config
model_config["compile"]["enabled"] = False
model_config["fixed_step_sampler_config"] = {
    "t_list": FIXED_STEP_SCHEDULE,
    "sample_type": "sde",
}
model_config["vlm_config_teacher"] = copy.deepcopy(model_config["vlm_config"])
model_config["vlm_config_fake_score"] = copy.deepcopy(model_config["vlm_config"])
model_config["teacher_load_from"] = {
    "load_path": "${oc.env:COSMOS_H_SURGICAL_DISTILL_TEACHER_DCP}/model",
    "credentials": "",
}
model_config["student_load_from"] = None
model_config["load_teacher_weights"] = True
model_config["simulation_mode"] = "backward"
model_config["backward_grad_steps"] = 1
model_config["teacher_guidance"] = 3.0
model_config["teacher_negative_prompt"] = ""
model_config["student_update_freq"] = 5
model_config["warmup_student_steps"] = 0
model_config["warmup_critic_steps"] = 0
model_config["vsd_gradient_space"] = "x0"
model_config["vsd_loss_reduction"] = "mean"
model_config["fake_score_loss_reduction"] = "active_mean"
model_config["loss_scale_sid"] = 1.0
model_config["loss_scale_fake_score"] = 1.0
model_config["grad_clip"] = True
model_config["rectified_flow_training_config"]["train_time_video_distribution"] = "uniform"

config = cosmos_h_surgical_transfer_dmd2_lora_480p_4step
config.trainer.type = DistillationTrainer
config.trainer.grad_accum_iter = 1
config.trainer.max_iter = 8000
config.trainer.callbacks = {
    "iter_speed": L(IterSpeed)(
        every_n="${trainer.logging_iter}",
        save_s3=False,
        hit_thres=0,
    ),
    "dmd2_metrics": L(DMD2Metrics)(
        output_path=(
            "${oc.env:IMAGINAIRE_OUTPUT_ROOT,/tmp/imaginaire4-output}/"
            "${job.project}/${job.group}/${job.name}/dmd2_metrics.jsonl"
        )
    ),
    "dmd2_wandb": L(DMD2WandBCallback)(),
}
config.optimizer = DMD2OptimizerConfig(
    net=_dmd2_optimizer(5.0e-5),
    fake_score=_dmd2_optimizer(5.0e-5),
)
config.scheduler.cycle_lengths = [8000]
config.scheduler.f_min = [0.1]
config.scheduler.warm_up_steps = [100]
config.checkpoint.load_path = ""
config.checkpoint.load_training_state = False
config.checkpoint.keys_to_skip_loading = []
config.checkpoint.save_iter = 50
config.checkpoint.strict_resume = True
config.dataloader_train = make_packing_dataloader(
    dataset_name="surgical_transfer",
    dataset=L(get_surgical_transfer_json_dataset)(
        **common_dataset_kwargs(),
        blur_suffix=".blur.mp4",
        control_modalities={
            "edge": 2.0,
            "blur": 2.0,
            "depth": 2.0,
            "seg": 1.0,
            "seg_tool": 1.0,
        },
        dataset_dir="${oc.env:COSMOS_H_SURGICAL_TRANSFER_DATASET_DIRS}",
        depth_suffix=".depth.mp4",
        enlarged_factor="${oc.env:COSMOS_H_SURGICAL_TRANSFER_ENLARGED_FACTORS,1.0}",
        json_path="${oc.env:COSMOS_H_SURGICAL_TRANSFER_JSON_PATHS}",
        seg_suffix=".seg.mp4",
        seg_tool_suffix=".seg_tool.mp4",
    ),
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=cosmos_h_surgical_transfer_dmd2_lora_480p_4step,
)
