# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_h_surgical.training import _install_distributed_affinity_compat, run_framework_cli


def test_distributed_affinity_compat_intersects_nvml_with_current_cpuset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        def __init__(self, local_rank: int) -> None:
            self.local_rank = local_rank

        def get_cpu_affinity(self) -> list[int]:
            return [0, 2, 4]

    distributed = SimpleNamespace(Device=FakeDevice, log=SimpleNamespace(warning=Mock()))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {2, 3})

    _install_distributed_affinity_compat(distributed)
    installed_device = distributed.Device
    device = installed_device(5)

    assert device.local_rank == 5
    assert device.get_cpu_affinity() == [2]
    _install_distributed_affinity_compat(distributed)
    assert distributed.Device is installed_device


def test_distributed_affinity_compat_keeps_cpuset_when_there_is_no_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        def __init__(self, _local_rank: int) -> None:
            pass

        def get_cpu_affinity(self) -> list[int]:
            return [0, 2, 4]

    warning = Mock()
    distributed = SimpleNamespace(Device=FakeDevice, log=SimpleNamespace(warning=warning))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {8, 9})

    _install_distributed_affinity_compat(distributed)

    assert distributed.Device(5).get_cpu_affinity() == [8, 9]
    warning.assert_called_once()


def test_framework_training_cli_registers_and_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    original = list(sys.argv)
    events: list[object] = []
    monkeypatch.delenv("COSMOS_TRAINING", raising=False)

    def fake_register() -> object:
        events.append("register")
        return object()

    def fake_install_compat() -> None:
        events.append("compat")

    def fake_run_module(module: str, *, run_name: str) -> dict[str, object]:
        events.append((module, run_name, list(sys.argv)))
        return {}

    assert (
        run_framework_cli(
            ["--sft-toml", "recipe.toml", "trainer.max_iter=1"],
            run_module=fake_run_module,
            register=fake_register,
            install_compat=fake_install_compat,
        )
        == 0
    )
    assert events == [
        "register",
        "compat",
        (
            "cosmos_framework.scripts.train",
            "__main__",
            ["cosmos-h-surgical train", "--sft-toml", "recipe.toml", "trainer.max_iter=1"],
        ),
    ]
    assert sys.argv == original
    assert os.environ["COSMOS_TRAINING"] == "1"


def test_training_dependency_error_is_actionable() -> None:
    def fake_register() -> object:
        return object()

    def missing_dependency(*args: object, **kwargs: object) -> dict[str, object]:
        raise ModuleNotFoundError("No module named 'webdataset'", name="webdataset")

    with pytest.raises(RuntimeError, match="uv sync --group cu130"):
        run_framework_cli([], run_module=missing_dependency, register=fake_register, install_compat=lambda: None)
