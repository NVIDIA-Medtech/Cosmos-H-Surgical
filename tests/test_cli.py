# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest

import cosmos_h_surgical.cli as cli
from cosmos_h_surgical.__about__ import __version__
from cosmos_h_surgical.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"cosmos-h-surgical {__version__}"


def test_framework_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["framework-info"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["repository"] == "https://github.com/NVIDIA/cosmos-framework.git"
    assert len(value["revision"]) == 40
    assert value["status"] == "audit-baseline"


def test_infer_forwards_framework_help(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def fake_framework_cli(argv: list[str]) -> int:
        observed.extend(argv)
        return 0

    monkeypatch.setattr(cli, "run_framework_cli", fake_framework_cli)
    assert main(["infer", "--help"]) == 0
    assert observed == ["--help"]


def test_train_forwards_framework_help(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def fake_framework_cli(argv: list[str]) -> int:
        observed.extend(argv)
        return 0

    monkeypatch.setattr(cli, "run_training_cli", fake_framework_cli)
    assert main(["train", "--help"]) == 0
    assert observed == ["--help"]


def test_prompt_upsample_forwards_framework_help(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[str] = []

    def fake_framework_cli(argv: list[str]) -> int:
        observed.extend(argv)
        return 0

    monkeypatch.setattr(cli, "run_prompt_upsampling_cli", fake_framework_cli)
    assert main(["prompt-upsample", "--help"]) == 0
    assert observed == ["--help"]
