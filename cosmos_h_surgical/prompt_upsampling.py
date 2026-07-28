# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from cosmos_framework.inference import prompt_upsampling as framework_prompt_upsampling

from cosmos_h_surgical.video_sampling import SampledVideo, sample_video_frames

_VIDEO_MODE = "video"
_VIDEO_MODES = frozenset({_VIDEO_MODE})
_IMAGE_MODES = frozenset({"image2video", "posttrain_image2video"})
log = logging.getLogger(__name__)


def _optional_int(value: str) -> int | None:
    if value.strip().lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer or 'none', got {value!r}") from error


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def _parser_action(parser: argparse.ArgumentParser, destination: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == destination:
            return action
    raise RuntimeError(f"Pinned Cosmos Framework prompt-upsampling CLI has no {destination!r} option")


def build_cli_parser() -> argparse.ArgumentParser:
    """Extend the pinned framework parser with surgical release features."""
    parser = framework_prompt_upsampling.build_cli_parser()

    mode_action = _parser_action(parser, "mode")
    mode_action.choices = [*mode_action.choices, _VIDEO_MODE]
    mode_action.help = "Prompt upsampling mode, including video for chronological source-video frames."

    top_k_action = _parser_action(parser, "top_k")
    top_k_action.type = _optional_int
    top_k_action.default = None
    top_k_action.help = "Optional sampling top-k, or 'none' (default: none)."

    parser.add_argument("--video", default=None, help="Shared local video path for video mode.")
    parser.add_argument(
        "--video-list",
        default=None,
        help="Text file with one local video path per prompt for video mode.",
    )
    parser.add_argument(
        "--video-frame-count",
        type=_positive_int,
        default=10,
        help="Minimum number of frames sampled across each source video (default: 10).",
    )
    parser.add_argument(
        "--video-min-fps",
        type=_positive_float,
        default=1.0,
        help="Minimum temporal sampling density in frames per second (default: 1.0).",
    )
    parser.add_argument(
        "--video-max-frames",
        type=_positive_int,
        default=32,
        help="Maximum frames allowed in one video-level request (default: 32).",
    )
    return parser


class _LoggingSession:
    def __init__(self, session: Any) -> None:
        self._session = session

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        payload = kwargs.get("json")
        model = payload.get("model") if isinstance(payload, dict) else None
        model_text = f" model={model}" if model else ""
        log.debug("HTTP %s %s%s", method, url, model_text)
        started = time.monotonic()
        try:
            response = self._session.request(method, url, **kwargs)
        except Exception:
            log.debug("HTTP %s %s failed after %.1fs", method, url, time.monotonic() - started)
            raise
        log.debug(
            "HTTP %s %s -> %s in %.1fs",
            method,
            url,
            response.status_code,
            time.monotonic() - started,
        )
        return response


class _ProgressPromptUpsamplerClient(framework_prompt_upsampling.PromptUpsamplerClient):
    def _with_retries(self, operation: str, fn: Callable[[], Any]) -> Any:
        if self.config.max_retries < 1:
            raise ValueError("max_retries must be >= 1.")
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            log.debug("%s attempt %d/%d", operation, attempt, self.config.max_retries)
            try:
                return fn()
            except Exception as error:
                last_error = error
                if attempt == self.config.max_retries:
                    break
                delay = self.config.retry_base_delay_s * (2 ** (attempt - 1))
                log.warning(
                    "%s attempt %d/%d failed: %s; retrying in %.1fs",
                    operation,
                    attempt,
                    self.config.max_retries,
                    error,
                    delay,
                )
                self._sleep(delay)
        raise RuntimeError(
            f"Prompt upsampler failed to {operation} after {self.config.max_retries} attempts: {last_error}"
        ) from last_error


def _make_client(config: Any) -> _ProgressPromptUpsamplerClient:
    session = framework_prompt_upsampling._make_session(config)
    return _ProgressPromptUpsamplerClient(config, session=_LoggingSession(session))


def _read_nonempty_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _apply_video_parameters(
    data: dict[str, Any],
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
) -> dict[str, Any]:
    if resolution not in framework_prompt_upsampling.RESOLUTION_RATIO_DICT:
        raise ValueError(f"Unsupported upsampler resolution {resolution!r}")
    if aspect_ratio not in framework_prompt_upsampling.RESOLUTION_RATIO_DICT[resolution]:
        raise ValueError(f"Unsupported aspect ratio {aspect_ratio!r} for resolution {resolution!r}")
    resolution_pair = framework_prompt_upsampling.RESOLUTION_RATIO_DICT[resolution][aspect_ratio]
    data["resolution"] = {"H": resolution_pair["H"], "W": resolution_pair["W"]}
    data["aspect_ratio"] = aspect_ratio
    data["duration"] = duration
    data["fps"] = fps
    return data


def _video_messages(
    prompt: str,
    sampled_video: SampledVideo,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
) -> list[dict[str, Any]]:
    timestamp_text = ", ".join(f"{value:.3f}s" for value in sampled_video.timestamps_seconds)
    video_note = (
        "IMPORTANT - VIDEO INPUT: The attached images are chronological frames sampled from one source video at "
        f"timestamps [{timestamp_text}]. Use all frames as visual and temporal ground truth for subjects, instruments, "
        "setting, lighting, colors, actions, and state changes. Describe the observed temporal evolution rather than "
        "treating the frames as unrelated images, and do not invent transitions contradicted by the frames.\n\n"
    )
    message_text = video_note + framework_prompt_upsampling.build_t2v_prompt_text(
        prompt,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration=duration,
        fps=fps,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": message_text}]
    for index, (data_url, timestamp) in enumerate(
        zip(sampled_video.data_urls, sampled_video.timestamps_seconds, strict=True),
        start=1,
    ):
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Chronological frame {index}/{len(sampled_video.data_urls)} at {timestamp:.3f}s:",
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
    return [framework_prompt_upsampling.SYSTEM_MESSAGE, {"role": "user", "content": content}]


def _upsample_video_prompt(
    client: Any,
    prompt: str,
    sampled_video: SampledVideo,
    *,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
) -> dict[str, Any]:
    messages = _video_messages(
        prompt,
        sampled_video,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration=duration,
        fps=fps,
    )
    raw_result = client.upsample_messages(messages)
    data = framework_prompt_upsampling.extract_json_object(raw_result)
    data = _apply_video_parameters(
        data,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration=duration,
        fps=fps,
    )
    return {"prompt": json.dumps(data, ensure_ascii=framework_prompt_upsampling.JSON_ENSURE_ASCII)}


def _write_record_atomically(output_directory: Path, index: int, record: dict[str, Any]) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"prompt_{index}.json"
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=output_directory,
            delete=False,
        ) as temporary:
            json.dump(record, temporary, ensure_ascii=framework_prompt_upsampling.JSON_ENSURE_ASCII, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _format_elapsed(seconds: float) -> str:
    rounded = max(0, round(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def upsample_prompt_file(
    client: Any,
    *,
    input_path: str | Path,
    output_path: str | Path,
    mode: str,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    image_url: str | None = None,
    image_list_path: str | Path | None = None,
    video: str | None = None,
    video_list_path: str | Path | None = None,
    video_frame_count: int = 10,
    video_min_fps: float = 1.0,
    video_max_frames: int = 32,
) -> list[dict[str, Any]]:
    prompts = _read_nonempty_lines(input_path)
    if not prompts:
        raise ValueError(f"No prompts found in {input_path}")

    image_urls = _read_nonempty_lines(image_list_path) if image_list_path is not None else None
    if image_urls is not None and len(image_urls) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} image entries in {image_list_path}, found {len(image_urls)}")

    videos = _read_nonempty_lines(video_list_path) if video_list_path is not None else None
    if videos is not None and len(videos) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} video entries in {video_list_path}, found {len(videos)}")

    output_directory = Path(output_path)
    results: list[dict[str, Any]] = []
    total = len(prompts)
    overall_started = time.monotonic()
    for index, prompt in enumerate(prompts):
        item_number = index + 1
        percent = round(item_number / total * 100)
        item_started = time.monotonic()
        log.info("[%d/%d %d%%] Processing prompt", item_number, total, percent)

        if mode in _VIDEO_MODES:
            current_video = videos[index] if videos is not None else video
            if current_video is None:
                raise ValueError("video mode requires --video or --video-list")
            sampled_video = sample_video_frames(
                current_video,
                frame_count=video_frame_count,
                min_fps=video_min_fps,
                max_frames=video_max_frames,
            )
            log.info(
                "[%d/%d %d%%] Sampled %d frames from %s (%.2fs, source %.2f FPS, effective %.2f FPS)",
                item_number,
                total,
                percent,
                len(sampled_video.data_urls),
                Path(current_video).name,
                sampled_video.duration_seconds,
                sampled_video.source_fps,
                sampled_video.effective_sample_fps,
            )
            log.debug(
                "[%d/%d] Sample timestamps: %s",
                item_number,
                total,
                ", ".join(f"{value:.3f}s" for value in sampled_video.timestamps_seconds),
            )
            record = _upsample_video_prompt(
                client,
                prompt,
                sampled_video,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
            )
        else:
            current_image_url = image_urls[index] if image_urls is not None else image_url
            record = framework_prompt_upsampling._upsample_prompt_for_mode(
                client,
                prompt,
                mode=mode,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                fps=fps,
                image_url=current_image_url,
            )

        destination = _write_record_atomically(output_directory, index, record)
        results.append(record)
        elapsed = time.monotonic() - overall_started
        average = elapsed / item_number
        eta = average * (total - item_number)
        log.info(
            "[%d/%d %d%%] Wrote %s (item %.1fs, elapsed %s, ETA %s)",
            item_number,
            total,
            percent,
            destination,
            time.monotonic() - item_started,
            _format_elapsed(elapsed),
            _format_elapsed(eta),
        )

    log.info("Completed %d/%d prompts in %s", total, total, _format_elapsed(time.monotonic() - overall_started))
    return results


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.endpoint_url is None:
        parser.error("--endpoint-url is required unless PROMPT_UPSAMPLER_ENDPOINT_URL is set.")
    if args.image_url is not None and args.image_list is not None:
        parser.error("Pass only one of --image-url or --image-list.")
    if args.video is not None and args.video_list is not None:
        parser.error("Pass only one of --video or --video-list.")
    if args.mode == _VIDEO_MODE:
        if args.video is None and args.video_list is None:
            parser.error("video mode requires --video or --video-list.")
        if args.image_url is not None or args.image_list is not None:
            parser.error("video mode does not accept --image-url or --image-list.")
    elif args.video is not None or args.video_list is not None:
        parser.error("--video and --video-list require --mode video.")
    if args.mode in _IMAGE_MODES and args.image_url is None and args.image_list is None:
        parser.error(f"{args.mode} mode requires --image-url or --image-list.")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    # The pinned client logs a token suffix at DEBUG. Our wrapper emits the
    # useful HTTP metadata itself without exposing any credential fragment.
    framework_prompt_upsampling.log.setLevel(logging.INFO)


def run_framework_cli(argv: Sequence[str]) -> int:
    """Run the pinned prompt upsampler with release-specific compatibility."""
    parser = build_cli_parser()
    args = parser.parse_args(list(argv))
    _configure_logging(args.verbose)
    _validate_cli_args(parser, args)

    template_mode = "image2video" if args.mode == _VIDEO_MODE else args.mode
    framework_prompt_upsampling.configure_prompting_templates(
        mode=template_mode,
        prompt_template_path=args.prompt_template,
        json_template_path=args.json_template,
    )
    config = framework_prompt_upsampling.PromptUpsamplerConfig(
        endpoint_url=args.endpoint_url,
        model=args.model,
        api_token=args.api_token,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_base_delay_s=args.retry_base_delay_s,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
    )
    client = _make_client(config)
    upsample_prompt_file(
        client,
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        duration=args.duration,
        fps=args.fps,
        image_url=args.image_url,
        image_list_path=args.image_list,
        video=args.video,
        video_list_path=args.video_list,
        video_frame_count=args.video_frame_count,
        video_min_fps=args.video_min_fps,
        video_max_frames=args.video_max_frames,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_framework_cli(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
