# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

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
    assert value["status"] == "pinned-release"


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


def test_prepare_training_data_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    video = dataset / "clip.mp4"
    video.touch()
    video.with_suffix(".json").write_text(json.dumps({"caption_json": {"description": "Action."}}))
    output = dataset / "train.json"

    assert (
        main(
            [
                "prepare-training-data",
                "--dataset-dir",
                str(dataset),
                "--video-pattern",
                "*.mp4",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert "Wrote 1 target videos" in capsys.readouterr().out


def test_validate_training_data_command_without_media_probe(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    video = dataset / "clip.mp4"
    video.touch()
    video.with_suffix(".json").write_text(json.dumps({"caption_json": {"description": "Action."}}))
    manifest = dataset / "train.json"
    manifest.write_text(json.dumps({"training": ["clip.mp4"]}))

    assert (
        main(
            [
                "validate-training-data",
                "--mode",
                "predict",
                "--dataset-dir",
                str(dataset),
                "--manifest",
                str(manifest),
                "--skip-media-probe",
            ]
        )
        == 0
    )
    assert "Validated 1 videos" in capsys.readouterr().out
