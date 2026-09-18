# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Surgical transfer compatibility layer for Cosmos Framework DMD2."""

from __future__ import annotations

from typing import Any, cast

import torch
from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.model.generator.distillation.common_loss import (
    variational_score_distillation_loss,
    variational_score_distillation_loss_from_gradient,
)
from cosmos_framework.model.generator.distillation.dmd2_rf import DMD2RFModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.model.generator.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
)
from cosmos_framework.utils import log
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict

_ACTION_ONLY_CHECKPOINT_PREFIXES = ("action2llm.", "llm2action.", "action_modality_embed")


def generated_vision_item_indices(
    num_vision_items_per_sample: list[int] | None,
    num_vision_items: int,
) -> list[int]:
    """Return the final (generated target) vision item for every sample."""
    if num_vision_items_per_sample is None:
        return list(range(num_vision_items))
    if sum(num_vision_items_per_sample) != num_vision_items:
        raise ValueError(
            "num_vision_items_per_sample must sum to the number of flat vision items, "
            f"got counts={num_vision_items_per_sample} and {num_vision_items} items."
        )

    indices: list[int] = []
    offset = 0
    for count in num_vision_items_per_sample:
        if count <= 0:
            raise ValueError(f"Vision item counts must be positive, got {num_vision_items_per_sample}.")
        indices.append(offset + count - 1)
        offset += count
    return indices


def _select_items(items: list[Any], indices: list[int]) -> list[Any]:
    return [items[index] for index in indices]


class SurgicalDMD2RFModel(DMD2RFModel):
    """DMD2 with release-LoRA teachers and target-only transfer losses.

    The released Surgical checkpoint is Nano plus a trained LoRA adapter. The
    teacher, student, and fake-score networks must therefore all be built with
    LoRA, initialized from the same released state, and only then frozen or
    optimized according to their DMD2 role.
    """

    def __init__(self, config: Any) -> None:
        if not config.lora_enabled:
            raise ValueError("Surgical DMD2 requires the released Surgical LoRA architecture.")
        if config.action_gen or config.sound_gen:
            raise ValueError("The initial Surgical DMD2 implementation supports vision transfer only.")
        if config.simulation_mode != "backward":
            raise ValueError("Surgical DMD2 requires backward fixed-step simulation.")
        if config.fixed_step_sampler_config.sample_type != "sde":
            raise ValueError("Surgical DMD2 currently requires the validated SDE sampler.")
        super().__init__(config)

    def build_net(self, dtype: torch.dtype, *, lora_enabled: bool | None = None) -> torch.nn.Module:
        """Keep LoRA enabled when the pinned Framework constructs the teacher."""
        if self.config.lora_enabled and lora_enabled is False:
            lora_enabled = True
        return super().build_net(dtype, lora_enabled=lora_enabled)

    def _copy_teacher_weights(self, target_net: torch.nn.Module, target_name: str) -> None:
        """Clone the full released base-plus-LoRA state without silent gaps."""
        teacher_state = {k: v for k, v in self.net_teacher.state_dict().items() if not k.endswith("_extra_state")}
        target_state = {k: v for k, v in target_net.state_dict().items() if not k.endswith("_extra_state")}
        missing = sorted(set(target_state) - set(teacher_state))
        unexpected = sorted(set(teacher_state) - set(target_state))
        lora_keys = sorted(key for key in teacher_state if ".lora_" in key)
        if not lora_keys:
            raise RuntimeError("The teacher has no LoRA tensors; refusing to initialize Surgical DMD2.")
        if missing or unexpected:
            raise RuntimeError(
                f"teacher -> {target_name} architecture mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        result = target_net.load_state_dict(teacher_state, strict=False)
        bad_missing = [key for key in result.missing_keys if not key.endswith("_extra_state")]
        bad_unexpected = [key for key in result.unexpected_keys if not key.endswith("_extra_state")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"teacher -> {target_name} load mismatch: missing={bad_missing[:20]}, unexpected={bad_unexpected[:20]}"
            )
        log.info(f"Copied {len(teacher_state)} tensors, including {len(lora_keys)} LoRA tensors, to {target_name}.")

    def _load_checkpoint_to_net(
        self,
        net: torch.nn.Module,
        ckpt_path: str,
        prefix: str = "net_ema",
        credential_path: str | None = None,
    ) -> None:
        """Audit transfer-relevant DCP keys before the Framework partial load."""
        storage_reader = (
            S3StorageReader(credential_path=credential_path or "", path=ckpt_path)
            if ckpt_path.startswith("s3://")
            else FileSystemReader(ckpt_path)
        )
        checkpoint_keys = set(storage_reader.read_metadata().state_dict_metadata)
        prefixes = [
            candidate
            for candidate in ("net", "net_ema")
            if any(key.startswith(f"{candidate}.") for key in checkpoint_keys)
        ]
        if len(prefixes) != 1:
            raise RuntimeError(f"Expected exactly one model prefix in teacher DCP, found {prefixes}.")
        prefix = prefixes[0]

        model_keys = {key for key in get_model_state_dict(net) if not key.endswith("_extra_state")}
        expected = {f"{prefix}.{key}" for key in model_keys}
        missing = sorted(expected - checkpoint_keys)
        checkpoint_model_keys = {
            key for key in checkpoint_keys if key.startswith(f"{prefix}.") and not key.endswith("_extra_state")
        }
        unexpected = sorted(checkpoint_model_keys - expected)
        bad_unexpected = [
            key for key in unexpected if not key.removeprefix(f"{prefix}.").startswith(_ACTION_ONLY_CHECKPOINT_PREFIXES)
        ]
        expected_lora = {key for key in expected if ".lora_" in key}
        if not expected_lora:
            raise RuntimeError("The Surgical teacher network exposes no LoRA keys.")
        if missing or bad_unexpected:
            raise RuntimeError(
                "Teacher DCP is incompatible with the transfer-only Surgical model: "
                f"missing={missing[:20]}, unexpected_non_action={bad_unexpected[:20]}"
            )
        log.info(
            f"Validated teacher DCP: {len(expected)} transfer tensors, {len(expected_lora)} LoRA tensors, "
            f"{len(unexpected)} allowed action-only tensor(s)."
        )
        super()._load_checkpoint_to_net(net, ckpt_path, prefix=prefix, credential_path=credential_path)

    def _get_vision_noise_sampling_metadata(
        self,
        data_batch: dict[str, Any],
        gen_data_clean: GenerationDataClean,
    ) -> tuple[list[str], list[int]]:
        resolutions, _ = super()._get_vision_noise_sampling_metadata(data_batch, gen_data_clean)
        assert gen_data_clean.x0_tokens_vision is not None
        indices = generated_vision_item_indices(
            gen_data_clean.num_vision_items_per_sample,
            len(gen_data_clean.x0_tokens_vision),
        )
        tokens_per_item = [item.shape[2] * item.shape[3] * item.shape[4] for item in gen_data_clean.x0_tokens_vision]
        return resolutions, [tokens_per_item[index] for index in indices]

    def _pack_and_denoise(
        self,
        gen_data_clean: GenerationDataClean,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        net_type: str,
    ) -> dict[str, torch.Tensor | object]:
        """Preserve transfer metadata while packing a noised DMD2 pass."""
        gen_data_for_packing = GenerationDataClean(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=[item.cpu() for item in gen_data_noised.xt_tokens_vision],
            fps_vision=gen_data_clean.fps_vision,
            temporal_positions_vision=gen_data_clean.temporal_positions_vision,
            num_vision_items_per_sample=gen_data_clean.num_vision_items_per_sample,
            control_weights=gen_data_clean.control_weights,
        )
        packed_sequence = self._pack_input_sequence(
            sequence_plans, input_text_indexes, gen_data_for_packing, timesteps.cpu()
        )
        assert packed_sequence.vision is not None, "Packed vision data is required"
        packed_sequence.vision.tokens = gen_data_noised.xt_tokens_vision
        packed_sequence.to_cuda()
        return self.denoise(net=self.denoiser_nets[net_type], data_batch_packed=packed_sequence)

    def _backward_simulation(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        iteration: int,
    ) -> tuple[GenerationDataClean, PackedSequence, torch.Tensor, dict]:
        """Run fixed-step SDE simulation over every flat transfer vision item."""
        backward_grad_steps = self.config.backward_grad_steps
        if backward_grad_steps != -1 and backward_grad_steps < 1:
            raise ValueError(f"backward_grad_steps must be -1 or >= 1, got {backward_grad_steps}")
        assert gen_data_clean.x0_tokens_vision is not None

        batch_size = gen_data_clean.batch_size
        counts = gen_data_clean.num_vision_items_per_sample
        generated_vision_item_indices(counts, len(gen_data_clean.x0_tokens_vision))
        num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
        schedule = list(self.config.fixed_step_sampler_config.t_list)
        if not schedule or abs(schedule[0] - 1.0) >= 1e-6:
            raise ValueError(f"Backward simulation requires a pure-noise sigma 1.0 start, got {schedule}.")
        if schedule[-1] == 0.0:
            schedule = schedule[:-1]
        n_steps = self._backward_n_steps(len(schedule), iteration)
        rollout = schedule[:n_steps] + [0.0]
        grad_steps = n_steps if backward_grad_steps == -1 else backward_grad_steps

        sigma_max = torch.full((batch_size, 1), rollout[0], **self.tensor_kwargs_fp32)
        timesteps_max = sigma_max * num_train_timesteps
        packed_sequence = self._pack_input_sequence(
            sequence_plans, input_text_indexes, gen_data_clean, timesteps_max.cpu()
        )
        sigma_max_vision = _expand_per_sample_to_per_vision_item(sigma_max, counts)
        gen_data_noised = self._add_noise_to_input(
            gen_data_clean,
            packed_sequence,
            sigma_max_vision,
            sigmas_action=None,
            sigmas_sound=None,
        )
        assert packed_sequence.vision is not None
        condition_masks = cast(list[torch.Tensor], packed_sequence.vision.condition_mask)
        noisy_masks = [1.0 - mask for mask in condition_masks]
        num_items = len(noisy_masks)

        out_student: dict[str, Any] = {}
        x0_pred: list[torch.Tensor] = []
        for step, (sigma_value, sigma_next) in enumerate(zip(rollout[:-1], rollout[1:])):
            sigma = torch.full((batch_size, 1), sigma_value, **self.tensor_kwargs_fp32)
            sigma_vision = _expand_per_sample_to_per_vision_item(sigma, counts)
            timesteps = sigma * num_train_timesteps
            with torch.set_grad_enabled(step >= n_steps - grad_steps):
                out_student = self._pack_and_denoise(
                    gen_data_clean,
                    gen_data_noised,
                    timesteps,
                    input_text_indexes,
                    sequence_plans,
                    net_type="student",
                )

            xt_vision = [item.to(**self.tensor_kwargs) for item in gen_data_noised.xt_tokens_vision]
            masked_sigmas = [sigma_vision[index].view(1, 1, 1) * noisy_masks[index] for index in range(num_items)]
            x0_pred = self._velocity_to_x0(
                xt_vision,
                cast(list[torch.Tensor], out_student["preds_vision"]),
                masked_sigmas,
            )
            if sigma_next <= 0.0:
                continue

            if self.config.fixed_step_sampler_config.sample_type == "ode":
                xt_next = self._ode_step(
                    xt_vision,
                    cast(list[torch.Tensor], out_student["preds_vision"]),
                    noisy_masks,
                    sigma_next - sigma_value,
                )
            else:
                xt_next = self._sde_step(xt_vision, x0_pred, noisy_masks, condition_masks, sigma_next)
            sigma_next_tensor = torch.full((batch_size, 1), sigma_next, **self.tensor_kwargs_fp32)
            sigma_next_vision = _expand_per_sample_to_per_vision_item(sigma_next_tensor, counts)
            masked_next_sigmas = [
                sigma_next_vision[index].view(1, 1, 1) * noisy_masks[index] for index in range(num_items)
            ]
            gen_data_noised = GenerationDataNoised(
                batch_size=batch_size,
                epsilon_vision=gen_data_noised.epsilon_vision,
                xt_tokens_vision=xt_next,
                vt_target_vision=gen_data_noised.vt_target_vision,
                sigmas_vision=masked_next_sigmas,
            )

        generated = GenerationDataClean(
            batch_size=batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=x0_pred,
            fps_vision=gen_data_clean.fps_vision,
            temporal_positions_vision=gen_data_clean.temporal_positions_vision,
            num_vision_items_per_sample=counts,
            control_weights=gen_data_clean.control_weights,
        )
        return generated, packed_sequence, sigma_max, out_student

    def training_step_generator(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        data_resolutions: list[str],
        num_vision_tokens_per_sample: list[int],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Update the student using target-only transfer VSD."""
        batch_size = gen_data_clean.batch_size
        generated, packed_student, _, _ = self._gen_data_from_student(
            input_text_indexes, sequence_plans, gen_data_clean, iteration
        )
        assert generated.x0_tokens_vision is not None
        target_indices = generated_vision_item_indices(
            generated.num_vision_items_per_sample,
            len(generated.x0_tokens_vision),
        )
        latent_frames = [generated.x0_tokens_vision[index].shape[2] for index in target_indices]
        timesteps, sigmas = self._get_train_noise_level_vision(
            batch_size=batch_size,
            is_image_batch=generated.is_image_batch,
            num_vision_latent_frames=latent_frames,
            resolutions=data_resolutions,
            num_tokens=num_vision_tokens_per_sample,
        )
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.cp_mesh.get_group()
            source = torch.distributed.get_global_rank(cp_group, 0)
            torch.distributed.broadcast(timesteps.contiguous(), src=source, group=cp_group)
            torch.distributed.broadcast(sigmas.contiguous(), src=source, group=cp_group)

        vision_sigmas = _expand_per_sample_to_per_vision_item(sigmas, generated.num_vision_items_per_sample)
        noised = self._add_noise_to_input(
            generated,
            packed_student,
            vision_sigmas,
            sigmas_action=None,
            sigmas_sound=None,
        )
        with torch.no_grad():
            out_fake = self._pack_and_denoise(
                generated, noised, timesteps, input_text_indexes, sequence_plans, net_type="fake_score"
            )
            fake_v = cast(list[torch.Tensor], out_fake["preds_vision"])
            fake_x0 = self._velocity_to_x0(noised.xt_tokens_vision, fake_v, noised.sigmas_vision)

            out_teacher = self._pack_and_denoise(
                generated, noised, timesteps, input_text_indexes, sequence_plans, net_type="teacher"
            )
            teacher_v = cast(list[torch.Tensor], out_teacher["preds_vision"])
            teacher_v_guided = teacher_v
            teacher_x0 = self._velocity_to_x0(noised.xt_tokens_vision, teacher_v, noised.sigmas_vision)
            teacher_x0_cond = teacher_x0
            if self.config.teacher_guidance > 1.0:
                unconditional_text = [
                    tokenize_caption(
                        self.config.teacher_negative_prompt,
                        self.vlm_tokenizer,
                        is_video=not generated.is_image_batch,
                        use_system_prompt=self.vlm_config.use_system_prompt,
                    )
                    for _ in range(batch_size)
                ]
                out_unconditional = self._pack_and_denoise(
                    generated, noised, timesteps, unconditional_text, sequence_plans, net_type="teacher"
                )
                teacher_v_unconditional = cast(list[torch.Tensor], out_unconditional["preds_vision"])
                teacher_x0_unconditional = self._velocity_to_x0(
                    noised.xt_tokens_vision, teacher_v_unconditional, noised.sigmas_vision
                )
                guidance = self.config.teacher_guidance - 1.0
                teacher_v_guided = [
                    conditional + guidance * (conditional - unconditional)
                    for conditional, unconditional in zip(teacher_v, teacher_v_unconditional)
                ]
                teacher_x0 = [
                    conditional + guidance * (conditional - unconditional)
                    for conditional, unconditional in zip(teacher_x0, teacher_x0_unconditional)
                ]

        assert packed_student.vision is not None
        generated_x0 = _select_items(generated.x0_tokens_vision, target_indices)
        generated_teacher_x0 = _select_items(teacher_x0, target_indices)
        generated_teacher_x0_cond = _select_items(teacher_x0_cond, target_indices)
        generated_fake_x0 = _select_items(fake_x0, target_indices)
        generated_fake_v = _select_items(fake_v, target_indices)
        generated_teacher_v = _select_items(teacher_v_guided, target_indices)
        target_masks = [
            (1.0 - mask).to(generated_x0[0].dtype)
            for mask in _select_items(cast(list[Any], packed_student.vision.condition_mask), target_indices)
        ]
        if self.config.vsd_gradient_space == "velocity":
            vsd_gradient = [teacher - fake for teacher, fake in zip(generated_teacher_v, generated_fake_v)]
            vsd_loss = variational_score_distillation_loss_from_gradient(
                generated_x0,
                vsd_gradient,
                weight_reference=generated_teacher_x0,
                loss_mask=target_masks,
                reduction=self.config.vsd_loss_reduction,
            )
            raw_gradients = vsd_gradient
        elif self.config.vsd_gradient_space == "x0":
            vsd_loss = variational_score_distillation_loss(
                generated_x0,
                generated_teacher_x0,
                generated_fake_x0,
                loss_mask=target_masks,
                reduction=self.config.vsd_loss_reduction,
            )
            raw_gradients = [fake - teacher for fake, teacher in zip(generated_fake_x0, generated_teacher_x0)]
        else:
            raise ValueError(f"Unknown vsd_gradient_space={self.config.vsd_gradient_space}")
        total_loss = self.config.loss_scale_sid * vsd_loss

        with torch.no_grad():
            fake_teacher_diff = torch.stack(
                [(fake - teacher).abs().mean() for fake, teacher in zip(generated_fake_x0, generated_teacher_x0)]
            ).mean()
            gen_teacher_diff = torch.stack(
                [
                    (sample.detach() - teacher).abs().mean()
                    for sample, teacher in zip(generated_x0, generated_teacher_x0)
                ]
            ).mean()
            fake_student_diff = torch.stack(
                [(fake - sample.detach()).abs().mean() for fake, sample in zip(generated_fake_x0, generated_x0)]
            ).mean()
            fake_teacher_cond_diff = torch.stack(
                [(fake - teacher).abs().mean() for fake, teacher in zip(generated_fake_x0, generated_teacher_x0_cond)]
            ).mean()
            teacher_cfg_term = torch.stack(
                [
                    (guided - conditional).abs().mean()
                    for guided, conditional in zip(generated_teacher_x0, generated_teacher_x0_cond)
                ]
            ).mean()
            grad_norms = []
            for sample, teacher, gradient in zip(generated_x0, generated_teacher_x0, raw_gradients):
                normalizer = 1.0 / ((sample.detach().float() - teacher.float()).abs().mean() + 1e-6)
                grad_norms.append((gradient.float() * normalizer).norm())
            vsd_grad_norm = torch.stack(grad_norms).mean()
            denominator = gen_teacher_diff.clamp(min=1e-6)

        output = {
            "vsd_loss": vsd_loss.detach(),
            "total_generator_loss": total_loss.detach(),
            "sigma": sigmas.detach(),
            "flow_matching_loss_vision_per_instance": vsd_loss.detach().expand(batch_size),
            "dmd_loss_generator": total_loss.detach(),
            "dmd_loss": vsd_loss.detach(),
            "dmd_fake_teacher_diff": fake_teacher_diff.detach(),
            "dmd_fake_teacher_cond_diff": fake_teacher_cond_diff.detach(),
            "dmd_teacher_cfg_term": teacher_cfg_term.detach(),
            "dmd_gen_teacher_diff": gen_teacher_diff.detach(),
            "dmd_fake_student_diff": fake_student_diff.detach(),
            "dmd_fake_student_to_gen_teacher_ratio": (fake_student_diff / denominator).detach(),
            "dmd_fake_teacher_to_gen_teacher_ratio": (fake_teacher_diff / denominator).detach(),
            "dmd_vsd_grad_norm": vsd_grad_norm.detach(),
        }
        return output, total_loss

    def training_step_critic(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        data_resolutions: list[str],
        num_vision_tokens_per_sample: list[int],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Update fake score using generated target items, excluding controls."""
        batch_size = gen_data_clean.batch_size
        with torch.no_grad():
            generated, packed_student, _, _ = self._gen_data_from_student(
                input_text_indexes, sequence_plans, gen_data_clean, iteration
            )
        assert generated.x0_tokens_vision is not None
        target_indices = generated_vision_item_indices(
            generated.num_vision_items_per_sample,
            len(generated.x0_tokens_vision),
        )
        latent_frames = [generated.x0_tokens_vision[index].shape[2] for index in target_indices]
        timesteps, sigmas = self._get_train_noise_level_vision(
            batch_size=batch_size,
            is_image_batch=generated.is_image_batch,
            num_vision_latent_frames=latent_frames,
            resolutions=data_resolutions,
            num_tokens=num_vision_tokens_per_sample,
        )
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.cp_mesh.get_group()
            source = torch.distributed.get_global_rank(cp_group, 0)
            torch.distributed.broadcast(timesteps.contiguous(), src=source, group=cp_group)
            torch.distributed.broadcast(sigmas.contiguous(), src=source, group=cp_group)

        vision_sigmas = _expand_per_sample_to_per_vision_item(sigmas, generated.num_vision_items_per_sample)
        vision_timesteps = _expand_per_sample_to_per_vision_item(timesteps, generated.num_vision_items_per_sample)
        with torch.no_grad():
            noised = self._add_noise_to_input(
                generated,
                packed_student,
                vision_sigmas,
                sigmas_action=None,
                sigmas_sound=None,
            )
        out_fake = self._pack_and_denoise(
            generated, noised, timesteps, input_text_indexes, sequence_plans, net_type="fake_score"
        )
        assert packed_student.vision is not None
        reduction = self.config.fake_score_loss_reduction
        if reduction == "active_mean":
            rectified_flow = self.rectified_flow_image if generated.is_image_batch else self.rectified_flow_video
            _, per_item_loss = self._compute_flow_matching_loss(
                pred=out_fake["preds_vision"],
                target=noised.vt_target_vision,
                condition_mask=packed_student.vision.condition_mask,
                timesteps=vision_timesteps,
                has_valid_tokens=True,
                rectified_flow=rectified_flow,
                normalize_by_active=True,
            )
        elif reduction == "sum_rcm":
            _, per_item_loss = self._flow_matching_sum_rcm_loss(
                pred=out_fake["preds_vision"],
                target=noised.vt_target_vision,
                condition_mask=packed_student.vision.condition_mask,
                has_valid_tokens=True,
            )
        else:
            raise ValueError(f"Unknown fake-score loss reduction: {reduction}")
        if int(per_item_loss.shape[0]) != len(generated.x0_tokens_vision):
            raise ValueError(
                "Vision per-item loss must align with flat vision items, "
                f"got {int(per_item_loss.shape[0])} losses and {len(generated.x0_tokens_vision)} items."
            )
        target_loss = per_item_loss[target_indices]
        fake_score_loss = target_loss.mean() * self.config.loss_scale_fake_score
        output = {
            "fake_score_loss": fake_score_loss.detach(),
            "total_critic_loss": fake_score_loss.detach(),
            "sigma": sigmas.detach(),
            "flow_matching_loss_vision_per_instance": target_loss.detach(),
            "dmd_loss_critic": fake_score_loss.detach(),
            "dmd_loss": fake_score_loss.detach(),
        }
        return output, fake_score_loss
