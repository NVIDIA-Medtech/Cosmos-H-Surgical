# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cosmos_h_surgical.__about__ import __version__
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
