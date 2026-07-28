# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_h_surgical.video_sampling import calculate_sample_timestamps, sample_video_frames


def test_default_sampling_spans_short_video() -> None:
    timestamps = calculate_sample_timestamps(5.8125)
    assert len(timestamps) == 10
    assert timestamps[0] == pytest.approx(0.290625)
    assert timestamps[-1] == pytest.approx(5.521875)


def test_minimum_fps_increases_sample_count() -> None:
    timestamps = calculate_sample_timestamps(12.2)
    assert len(timestamps) == 13
    assert 12.2 / len(timestamps) <= 1.0


def test_maximum_frame_limit_fails_clearly() -> None:
    with pytest.raises(ValueError, match="exceeding max_frames=32"):
        calculate_sample_timestamps(32.1)


def test_real_sampling_contract_with_fake_decoder(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    observed: dict[str, Any] = {}

    class FakeDecoder:
        metadata = SimpleNamespace(
            begin_stream_seconds=0.0,
            end_stream_seconds=5.8125,
            duration_seconds=5.8125,
            average_fps=16.0,
        )

        def __len__(self) -> int:
            return 93

        def get_frames_played_at(self, timestamps: list[float]) -> Any:
            observed["timestamps"] = timestamps
            data = torch.zeros((len(timestamps), 32, 64, 3), dtype=torch.uint8)
            pts_seconds = torch.tensor(timestamps, dtype=torch.float64)
            return SimpleNamespace(data=data, pts_seconds=pts_seconds)

    def fake_decoder_factory(source: Path, **kwargs: Any) -> FakeDecoder:
        observed.update(source=source, kwargs=kwargs)
        return FakeDecoder()

    sampled = sample_video_frames(video, decoder_factory=fake_decoder_factory)

    assert observed["source"] == video
    assert observed["kwargs"] == {
        "dimension_order": "NHWC",
        "device": "cpu",
        "seek_mode": "exact",
    }
    assert len(sampled.data_urls) == 10
    assert all(value.startswith("data:image/jpeg;base64,") for value in sampled.data_urls)
    assert sampled.timestamps_seconds[0] == pytest.approx(0.290625)
    assert sampled.effective_sample_fps == pytest.approx(10 / 5.8125)


def test_source_fps_below_minimum_is_rejected(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    class FakeDecoder:
        metadata = SimpleNamespace(
            begin_stream_seconds=0.0,
            end_stream_seconds=5.0,
            duration_seconds=5.0,
            average_fps=0.5,
        )

        def __len__(self) -> int:
            return 3

    with pytest.raises(ValueError, match="below the requested minimum"):
        sample_video_frames(video, decoder_factory=lambda *_args, **_kwargs: FakeDecoder())
