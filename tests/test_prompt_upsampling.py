# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from cosmos_h_surgical import prompt_upsampling
from cosmos_h_surgical.video_sampling import SampledVideo


def _required_args(mode: str = "image2video") -> list[str]:
    return [
        "--input",
        "prompts.txt",
        "--output",
        "outputs",
        "--mode",
        mode,
        "--endpoint-url",
        "https://example.invalid",
        "--image-url",
        "image.jpg",
    ]


def test_top_k_is_omitted_by_default() -> None:
    args = prompt_upsampling.build_cli_parser().parse_args(_required_args())
    assert args.top_k is None


@pytest.mark.parametrize(("value", "expected"), [("none", None), ("20", 20)])
def test_top_k_accepts_none_or_integer(value: str, expected: int | None) -> None:
    args = prompt_upsampling.build_cli_parser().parse_args([*_required_args(), "--top-k", value])
    assert args.top_k == expected


def test_video_mode_parser_defaults() -> None:
    args = prompt_upsampling.build_cli_parser().parse_args(
        [
            "--input",
            "prompts.txt",
            "--output",
            "outputs",
            "--mode",
            "video",
            "--endpoint-url",
            "https://example.invalid",
            "--video",
            "video.mp4",
        ]
    )
    assert args.video_frame_count == 10
    assert args.video_min_fps == 1.0
    assert args.video_max_frames == 32


def test_default_cli_config_omits_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}

    def fake_make_client(config: Any) -> object:
        observed["config"] = config
        return object()

    monkeypatch.setattr(prompt_upsampling, "_make_client", fake_make_client)
    monkeypatch.setattr(
        prompt_upsampling.framework_prompt_upsampling,
        "configure_prompting_templates",
        lambda **_: None,
    )
    monkeypatch.setattr(prompt_upsampling, "upsample_prompt_file", lambda *_args, **_kwargs: [])

    assert prompt_upsampling.run_framework_cli(_required_args()) == 0
    assert observed["config"].top_k is None


def test_request_payload_omits_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    framework = prompt_upsampling.framework_prompt_upsampling
    config = framework.PromptUpsamplerConfig(
        endpoint_url="https://example.invalid",
        model="test-model",
        max_retries=1,
        top_k=None,
    )
    client = framework.PromptUpsamplerClient(config, session=object())
    observed: dict[str, Any] = {}

    def fake_request(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        observed.update(method=method, url=url, payload=payload)
        return {"choices": [{"message": {"content": '{"scene":"operating room"}'}}]}

    monkeypatch.setattr(client, "_request_json", fake_request)
    client.upsample_messages([{"role": "user", "content": "Describe the scene."}])

    assert observed["method"] == "POST"
    assert observed["url"] == "https://example.invalid/v1/chat/completions"
    assert "top_k" not in observed["payload"]


def test_video_messages_are_chronological() -> None:
    sampled_video = SampledVideo(
        data_urls=("data:image/jpeg;base64,first", "data:image/jpeg;base64,second"),
        timestamps_seconds=(0.25, 0.75),
        duration_seconds=1.0,
        source_fps=16.0,
    )
    messages = prompt_upsampling._video_messages(
        "short prompt",
        sampled_video,
        resolution="480",
        aspect_ratio="16,9",
        duration="6s",
        fps=16,
    )

    content = messages[1]["content"]
    assert "chronological frames" in content[0]["text"]
    assert content[1]["text"] == "Chronological frame 1/2 at 0.250s:"
    assert content[2]["image_url"]["url"].endswith("first")
    assert content[3]["text"] == "Chronological frame 2/2 at 0.750s:"
    assert content[4]["image_url"]["url"].endswith("second")


def test_video_prompt_applies_output_parameters() -> None:
    class FakeClient:
        def upsample_messages(self, messages: list[dict[str, Any]]) -> str:
            assert messages
            return '{"subjects":[]}'

    sampled_video = SampledVideo(
        data_urls=("data:image/jpeg;base64,frame",),
        timestamps_seconds=(0.5,),
        duration_seconds=1.0,
        source_fps=16.0,
    )
    record = prompt_upsampling._upsample_video_prompt(
        FakeClient(),
        "short prompt",
        sampled_video,
        resolution="480",
        aspect_ratio="16,9",
        duration="6s",
        fps=16,
    )
    structured_prompt = json.loads(record["prompt"])
    assert structured_prompt["resolution"] == {"H": 480, "W": 832}
    assert structured_prompt["aspect_ratio"] == "16,9"
    assert structured_prompt["duration"] == "6s"
    assert structured_prompt["fps"] == 16


def test_completed_records_survive_a_later_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first\nsecond\n", encoding="utf-8")
    output = tmp_path / "output"
    calls = 0

    def fake_upsample(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second request failed")
        return {"prompt": "{}"}

    monkeypatch.setattr(prompt_upsampling.framework_prompt_upsampling, "_upsample_prompt_for_mode", fake_upsample)
    with pytest.raises(RuntimeError, match="second request failed"):
        prompt_upsampling.upsample_prompt_file(
            object(),
            input_path=prompts,
            output_path=output,
            mode="image2video",
            resolution="480",
            aspect_ratio="16,9",
            duration="6s",
            fps=16,
            image_url="image.jpg",
        )

    assert json.loads((output / "prompt_0.json").read_text()) == {"prompt": "{}"}
    assert not (output / "prompt_1.json").exists()


def test_verbose_http_log_does_not_expose_token(caplog: pytest.LogCaptureFixture) -> None:
    class FakeResponse:
        status_code = 200

    class FakeSession:
        def request(self, *_args: Any, **_kwargs: Any) -> FakeResponse:
            return FakeResponse()

    caplog.set_level(logging.DEBUG, logger=prompt_upsampling.__name__)
    session = prompt_upsampling._LoggingSession(FakeSession())
    session.request(
        "POST",
        "https://example.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer REDACTED"},
        json={"model": "test-model"},
    )

    assert "HTTP POST" in caplog.text
    assert "-> 200" in caplog.text
    assert "REDACTED" not in caplog.text
