# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import sys

import pytest

from cosmos_h_surgical.training import run_framework_cli


def test_framework_training_cli_registers_and_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    original = list(sys.argv)
    events: list[object] = []
    monkeypatch.delenv("COSMOS_TRAINING", raising=False)

    def fake_register() -> object:
        events.append("register")
        return object()

    def fake_run_module(module: str, *, run_name: str) -> dict[str, object]:
        events.append((module, run_name, list(sys.argv)))
        return {}

    assert (
        run_framework_cli(
            ["--sft-toml", "recipe.toml", "trainer.max_iter=1"],
            run_module=fake_run_module,
            register=fake_register,
        )
        == 0
    )
    assert events == [
        "register",
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

    with pytest.raises(RuntimeError, match="uv sync --frozen --extra train"):
        run_framework_cli([], run_module=missing_dependency, register=fake_register)
