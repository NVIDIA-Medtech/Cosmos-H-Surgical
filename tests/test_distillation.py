# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from cosmos_framework.model.generator.distillation.dmd2_rf import DMD2RFModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

from cosmos_h_surgical.distillation import SurgicalDMD2RFModel, generated_vision_item_indices


class _TinyLoraNet(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.block = torch.nn.Module()
        self.block.weight = torch.nn.Parameter(torch.tensor([value]))
        self.block.lora_A = torch.nn.Parameter(torch.tensor([value + 1]))
        self.block.lora_B = torch.nn.Parameter(torch.tensor([value + 2]))


def _bare_model() -> SurgicalDMD2RFModel:
    return object.__new__(SurgicalDMD2RFModel)


def test_generated_vision_item_indices_selects_last_item_per_sample() -> None:
    assert generated_vision_item_indices(None, 3) == [0, 1, 2]
    assert generated_vision_item_indices([2, 2], 4) == [1, 3]
    assert generated_vision_item_indices([3, 1], 4) == [2, 3]
    with pytest.raises(ValueError, match="must sum"):
        generated_vision_item_indices([2, 2], 3)
    with pytest.raises(ValueError, match="positive"):
        generated_vision_item_indices([2, 0], 2)


def test_copy_teacher_weights_copies_released_lora_exactly() -> None:
    model = _bare_model()
    teacher = _TinyLoraNet(3.0)
    model._net_teacher_holder = [teacher]
    target = _TinyLoraNet(0.0)

    model._copy_teacher_weights(target, "student")

    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_copy_teacher_weights_rejects_adapter_mismatch() -> None:
    model = _bare_model()
    model._net_teacher_holder = [_TinyLoraNet(3.0)]
    target = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        model._copy_teacher_weights(target, "student")


def test_build_net_keeps_lora_enabled_for_teacher(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_model()
    model.config = SimpleNamespace(lora_enabled=True)
    calls: list[bool | None] = []

    def fake_build_net(
        _model: DMD2RFModel,
        _dtype: torch.dtype,
        *,
        lora_enabled: bool | None = None,
    ) -> torch.nn.Module:
        calls.append(lora_enabled)
        return _TinyLoraNet(0.0)

    monkeypatch.setattr(DMD2RFModel, "build_net", fake_build_net)

    model.build_net(torch.bfloat16, lora_enabled=False)

    assert calls == [True]


def test_teacher_dcp_audit_allows_only_release_action_surplus(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_model()
    net = _TinyLoraNet(0.0)
    checkpoint_keys = {
        "net.block.weight",
        "net.block.lora_A",
        "net.block.lora_B",
        "net.action2llm.fc.weight",
    }
    reader = SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata=dict.fromkeys(checkpoint_keys)))
    loaded: list[tuple[str, str]] = []

    monkeypatch.setattr("cosmos_h_surgical.distillation.FileSystemReader", lambda _path: reader)

    def fake_framework_load(
        _model: DMD2RFModel,
        _net: torch.nn.Module,
        checkpoint_path: str,
        prefix: str = "net_ema",
        credential_path: str | None = None,
    ) -> None:
        del credential_path
        loaded.append((checkpoint_path, prefix))

    monkeypatch.setattr(DMD2RFModel, "_load_checkpoint_to_net", fake_framework_load)

    model._load_checkpoint_to_net(net, "/teacher/model")

    assert loaded == [("/teacher/model", "net")]


def test_backward_rollout_expands_sample_sigmas_and_masks_controls() -> None:
    model = MagicMock()
    model.config = SimpleNamespace(
        backward_grad_steps=1,
        fixed_step_sampler_config=SimpleNamespace(t_list=[1.0], sample_type="sde"),
        rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
    )
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model._backward_n_steps.return_value = 1
    clean_items = [torch.zeros(1, 1, 1, 1) for _ in range(4)]
    clean = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=clean_items,
        num_vision_items_per_sample=[2, 2],
    )
    condition_masks = [
        torch.ones(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, 1),
        torch.zeros(1, 1, 1),
    ]
    packed = SimpleNamespace(vision=SimpleNamespace(condition_mask=condition_masks))
    model._pack_input_sequence.return_value = packed
    expanded_sigmas: list[torch.Tensor] = []

    def fake_add_noise(
        _clean: GenerationDataClean,
        _packed: object,
        sigmas: torch.Tensor,
        *,
        sigmas_action: torch.Tensor | None,
        sigmas_sound: torch.Tensor | None,
    ) -> SimpleNamespace:
        assert sigmas_action is None
        assert sigmas_sound is None
        expanded_sigmas.append(sigmas)
        return SimpleNamespace(
            epsilon_vision=[torch.zeros_like(item) for item in clean_items],
            xt_tokens_vision=[torch.zeros_like(item) for item in clean_items],
            vt_target_vision=[torch.zeros_like(item) for item in clean_items],
            sigmas_vision=[torch.zeros(1, 1, 1) for _ in clean_items],
        )

    model._add_noise_to_input.side_effect = fake_add_noise
    model._pack_and_denoise.return_value = {"preds_vision": [torch.zeros_like(item) for item in clean_items]}
    masked_sigmas: list[list[torch.Tensor]] = []

    def fake_velocity_to_x0(
        x_t: list[torch.Tensor],
        _velocity: list[torch.Tensor],
        sigmas: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        masked_sigmas.append(sigmas)
        return x_t

    model._velocity_to_x0.side_effect = fake_velocity_to_x0

    generated, returned_packed, _, _ = SurgicalDMD2RFModel._backward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=clean,
        iteration=0,
    )

    torch.testing.assert_close(expanded_sigmas[0], torch.ones(4, 1))
    assert [sigma.item() for sigma in masked_sigmas[0]] == [0.0, 1.0, 0.0, 1.0]
    assert generated.num_vision_items_per_sample == [2, 2]
    assert returned_packed is packed


def test_vision_noise_metadata_uses_transfer_targets_only() -> None:
    model = _bare_model()
    model.config = SimpleNamespace(resolution="480")
    gen_data = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=[
            torch.zeros(1, 1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2, 2),
            torch.zeros(1, 1, 3, 2, 2),
            torch.zeros(1, 1, 4, 2, 2),
        ],
        num_vision_items_per_sample=[2, 2],
    )
    resolutions, tokens = model._get_vision_noise_sampling_metadata({}, gen_data)
    assert resolutions == ["480", "480"]
    assert tokens == [8, 16]


def test_pack_and_denoise_preserves_transfer_metadata() -> None:
    model = _bare_model()
    temporal_positions = [torch.arange(2), torch.arange(2)]
    clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        raw_state_vision=[torch.zeros(1)],
        x0_tokens_vision=[torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)],
        fps_vision=torch.tensor([16.0]),
        temporal_positions_vision=temporal_positions,
        num_vision_items_per_sample=[2],
        control_weights=[[2.0]],
    )
    noised = SimpleNamespace(
        xt_tokens_vision=[torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)],
    )
    packed = SimpleNamespace(vision=SimpleNamespace(tokens=None), to_cuda=MagicMock())
    model._pack_input_sequence = MagicMock(return_value=packed)
    model.denoiser_nets = {"student": object()}
    model.denoise = MagicMock(return_value={"preds_vision": []})

    model._pack_and_denoise(clean, noised, torch.ones(1, 1), [[]], [MagicMock()], "student")

    proxy = model._pack_input_sequence.call_args.args[2]
    assert proxy.temporal_positions_vision is temporal_positions
    assert proxy.num_vision_items_per_sample == [2]
    assert proxy.control_weights == [[2.0]]


def test_generator_reduces_vsd_over_transfer_targets_only(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    model.config = SimpleNamespace(
        loss_scale_sid=1.0,
        teacher_guidance=1.0,
        vsd_gradient_space="x0",
        vsd_loss_reduction="mean",
    )
    model.parallel_dims = None
    generated_items = [torch.full((1, 1, 1, 1), value, requires_grad=True) for value in (100.0, 1.0, 200.0, 2.0)]
    generated = SimpleNamespace(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=generated_items,
        num_vision_items_per_sample=[2, 2],
    )
    packed = SimpleNamespace(vision=SimpleNamespace(condition_mask=[torch.zeros(1, 1, 1) for _ in range(4)]))
    noised = SimpleNamespace(
        xt_tokens_vision=[torch.zeros(1, 1, 1, 1) for _ in range(4)],
        sigmas_vision=[torch.full((1, 1, 1), 0.5) for _ in range(4)],
    )
    fake_x0 = [torch.full((1, 1, 1, 1), value) for value in (1000.0, 3.0, 2000.0, 4.0)]
    teacher_x0 = [torch.full((1, 1, 1, 1), value) for value in (3000.0, 5.0, 4000.0, 6.0)]
    model._gen_data_from_student.return_value = (generated, packed, torch.ones(2, 1), {})
    model._get_train_noise_level_vision.return_value = (torch.ones(2, 1), torch.full((2, 1), 0.5))
    model._add_noise_to_input.return_value = noised
    model._pack_and_denoise.side_effect = [
        {"preds_vision": [torch.zeros_like(item) for item in generated_items]},
        {"preds_vision": [torch.zeros_like(item) for item in generated_items]},
    ]
    model._velocity_to_x0.side_effect = [fake_x0, teacher_x0]
    captured: dict[str, list[torch.Tensor]] = {}

    def fake_vsd_loss(
        gen_data: list[torch.Tensor],
        teacher: list[torch.Tensor],
        fake: list[torch.Tensor],
        *,
        loss_mask: list[torch.Tensor],
        reduction: str,
    ) -> torch.Tensor:
        captured.update(gen_data=gen_data, teacher=teacher, fake=fake, loss_mask=loss_mask)
        assert reduction == "mean"
        return sum(item.sum() for item in gen_data)

    monkeypatch.setattr("cosmos_h_surgical.distillation.variational_score_distillation_loss", fake_vsd_loss)

    output, loss = SurgicalDMD2RFModel.training_step_generator(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=SimpleNamespace(batch_size=2),
        data_resolutions=["480", "480"],
        num_vision_tokens_per_sample=[1, 1],
        iteration=0,
    )

    assert [item.item() for item in captured["gen_data"]] == [1.0, 2.0]
    assert [item.item() for item in captured["teacher"]] == [5.0, 6.0]
    assert [item.item() for item in captured["fake"]] == [3.0, 4.0]
    assert len(captured["loss_mask"]) == 2
    torch.testing.assert_close(loss, torch.tensor(3.0))
    assert output["flow_matching_loss_vision_per_instance"].shape == (2,)


def test_critic_reduces_transfer_loss_over_targets_only() -> None:
    model = MagicMock()
    model.config = SimpleNamespace(fake_score_loss_reduction="active_mean", loss_scale_fake_score=1.0)
    model.parallel_dims = None
    generated = SimpleNamespace(
        batch_size=2,
        is_image_batch=False,
        x0_tokens_vision=[torch.zeros(1, 1, 1, 1) for _ in range(4)],
        num_vision_items_per_sample=[2, 2],
    )
    packed = SimpleNamespace(vision=SimpleNamespace(condition_mask=[torch.zeros(1, 1, 1) for _ in range(4)]))
    noised = SimpleNamespace(vt_target_vision=[torch.zeros(1, 1, 1, 1) for _ in range(4)])
    model._gen_data_from_student.return_value = (generated, packed, torch.ones(2, 1), {})
    model._get_train_noise_level_vision.return_value = (torch.ones(2, 1), torch.full((2, 1), 0.5))
    model._add_noise_to_input.return_value = noised
    model._pack_and_denoise.return_value = {"preds_vision": [torch.zeros(1, 1, 1, 1) for _ in range(4)]}
    model._compute_flow_matching_loss.return_value = (
        torch.tensor(0.0),
        torch.tensor([100.0, 2.0, 200.0, 4.0]),
    )
    model.rectified_flow_video = object()

    output, loss = SurgicalDMD2RFModel.training_step_critic(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=SimpleNamespace(batch_size=2),
        data_resolutions=["480", "480"],
        num_vision_tokens_per_sample=[1, 1],
        iteration=0,
    )

    torch.testing.assert_close(loss, torch.tensor(3.0))
    torch.testing.assert_close(output["flow_matching_loss_vision_per_instance"], torch.tensor([2.0, 4.0]))
