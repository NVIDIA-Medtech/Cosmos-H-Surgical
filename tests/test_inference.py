# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import sys

import pytest

from cosmos_h_surgical.inference import run_framework_cli


def test_framework_cli_forwards_arguments_and_restores_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    original = list(sys.argv)
    observed: list[str] = []
    monkeypatch.delenv("COSMOS_TRAINING", raising=False)

    def fake_entrypoint() -> None:
        observed.extend(sys.argv)

    assert run_framework_cli(["-i", "sample.json", "--seed", "0"], entrypoint=fake_entrypoint) == 0
    assert observed == ["cosmos-h-surgical infer", "-i", "sample.json", "--seed", "0"]
    assert sys.argv == original
    assert os.environ["COSMOS_TRAINING"] == "0"
