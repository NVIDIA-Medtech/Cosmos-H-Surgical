# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cosmos_h_surgical.__about__ import __version__
from cosmos_h_surgical.data.manifests import prepare_training_manifest, validate_training_manifests
from cosmos_h_surgical.inference import run_framework_cli
from cosmos_h_surgical.prompt_upsampling import run_framework_cli as run_prompt_upsampling_cli
from cosmos_h_surgical.provenance import FRAMEWORK_REPOSITORY, FRAMEWORK_REVISION, FRAMEWORK_STATUS
from cosmos_h_surgical.release import validate_manifest
from cosmos_h_surgical.training import run_framework_cli as run_training_cli


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosmos-h-surgical")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "infer",
        add_help=False,
        help="Run the pinned Cosmos Framework inference CLI",
    )
    subparsers.add_parser(
        "train",
        add_help=False,
        help="Run the pinned Cosmos Framework training CLI with surgical configs",
    )
    subparsers.add_parser(
        "prompt-upsample",
        add_help=False,
        help="Generate structured Cosmos 3 prompts with OpenAI-compatible defaults",
    )
    subparsers.add_parser("framework-info", help="Print immutable framework provenance")

    prepare_parser = subparsers.add_parser(
        "prepare-training-data",
        help="Build a portable development-format training manifest",
    )
    prepare_parser.add_argument("--dataset-dir", type=Path, required=True, help="Root containing target videos")
    prepare_parser.add_argument(
        "--video-pattern",
        action="append",
        dest="video_patterns",
        required=True,
        help="Recursive glob relative to --dataset-dir; repeat to add patterns",
    )
    prepare_parser.add_argument("--output", type=Path, required=True, help="Manifest JSON to write")
    prepare_parser.add_argument("--caption-format", choices=("json", "text"), default="json")
    prepare_parser.add_argument("--allow-missing-captions", action="store_true")
    prepare_parser.add_argument("--force", action="store_true", help="Replace an existing output manifest")

    validate_data_parser = subparsers.add_parser(
        "validate-training-data",
        help="Validate surgical manifests, captions, and transfer controls",
    )
    validate_data_parser.add_argument("--dataset-dir", type=Path, required=True)
    validate_data_parser.add_argument("--manifest", type=Path, action="append", required=True)
    validate_data_parser.add_argument("--mode", choices=("predict", "transfer"), required=True)
    validate_data_parser.add_argument("--caption-format", choices=("json", "text"), default="json")
    validate_data_parser.add_argument(
        "--control-modalities",
        default="edge,blur,depth,seg",
        help="Comma-separated transfer controls (default: edge,blur,depth,seg)",
    )
    validate_data_parser.add_argument("--minimum-frames", type=int, default=93)
    validate_data_parser.add_argument("--allow-absolute-paths", action="store_true")
    validate_data_parser.add_argument("--skip-media-probe", action="store_true")

    validate_parser = subparsers.add_parser("validate-release", help="Validate public release metadata")
    validate_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release-manifest.json"),
        help="Release manifest to validate (default: ./release-manifest.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, remainder = parser.parse_known_args(argv)

    if args.command == "infer":
        return run_framework_cli(remainder or ["--help"])
    if args.command == "train":
        return run_training_cli(remainder or ["--help"])
    if args.command == "prompt-upsample":
        return run_prompt_upsampling_cli(remainder or ["--help"])
    if remainder:
        parser.error(f"unrecognized arguments: {' '.join(remainder)}")
    if args.command == "prepare-training-data":
        try:
            count = prepare_training_manifest(
                args.dataset_dir,
                args.output,
                args.video_patterns,
                caption_format=args.caption_format,
                require_captions=not args.allow_missing_captions,
                force=args.force,
            )
        except (FileExistsError, ValueError) as error:
            print(f"ERROR: {error}")
            return 1
        print(f"Wrote {count} target videos to {args.output}")
        return 0
    if args.command == "validate-training-data":
        modalities = [value.strip() for value in args.control_modalities.split(",") if value.strip()]
        validation_options = {"media_probe": None} if args.skip_media_probe else {}
        try:
            report = validate_training_manifests(
                args.dataset_dir,
                args.manifest,
                mode=args.mode,
                caption_format=args.caption_format,
                control_modalities=modalities,
                minimum_frames=args.minimum_frames,
                require_relative=not args.allow_absolute_paths,
                **validation_options,
            )
        except ValueError as error:
            print(f"ERROR: {error}")
            return 1
        if report.errors:
            for error in report.errors:
                print(f"ERROR: {error}")
            return 1
        print(f"Validated {report.videos} videos across {report.manifests} manifest(s)")
        return 0
    if args.command == "framework-info":
        print(
            json.dumps(
                {
                    "repository": FRAMEWORK_REPOSITORY,
                    "revision": FRAMEWORK_REVISION,
                    "status": FRAMEWORK_STATUS,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-release":
        errors = validate_manifest(args.manifest)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(f"Validated {args.manifest}")
        return 0

    parser.print_help()
    return 0
