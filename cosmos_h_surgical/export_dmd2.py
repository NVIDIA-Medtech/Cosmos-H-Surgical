# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export a DMD2 training checkpoint as a portable student-only model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cosmos_h_surgical.checkpoints import BASE_MODEL_CONFIG_PATH

_EXPECTED_FIXED_STEP_SAMPLER = {
    "sample_type": "sde",
    "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
}
_FORBIDDEN_PUBLIC_MARKERS = tuple("/" + directory for directory in ("home/", "workspace/", "lustre/", "healthcareeng_"))


def _nested(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = config
    for key in keys:
        value = value[key]
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary at {'.'.join(keys)}")
    return value


def sanitize_student_export_config(output_dir: Path) -> None:
    """Replace run-local paths in an exported student with public aliases."""
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_config = json.loads(BASE_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = _nested(config, "model", "config")
    public_model_config = _nested(base_config, "model", "config")

    fixed_step_sampler = model_config.get("fixed_step_sampler_config")
    if fixed_step_sampler != _EXPECTED_FIXED_STEP_SAMPLER:
        raise ValueError(f"Unexpected DMD2 fixed-step sampler: {fixed_step_sampler!r}")

    tokenizer = _nested(model_config, "tokenizer")
    public_tokenizer = _nested(public_model_config, "tokenizer")
    tokenizer["bucket_name"] = public_tokenizer["bucket_name"]
    tokenizer["vae_path"] = public_tokenizer["vae_path"]
    tokenizer["object_store_credential_path_pretrained"] = ""

    vlm_config = _nested(model_config, "vlm_config")
    pretrained_weights = _nested(vlm_config, "pretrained_weights")
    pretrained_weights.update(
        {
            "backbone_path": "",
            "credentials_path": "",
            "enable_gcs_patch_in_boto3": False,
            "enabled": False,
        }
    )
    base_config_node = _nested(vlm_config, "model_instance", "config", "base_config")
    public_base_config_node = _nested(public_model_config, "vlm_config", "model_instance", "config", "base_config")
    base_config_node["json_file"] = public_base_config_node["json_file"]

    serialized = json.dumps(config, indent=2) + "\n"
    for marker in _FORBIDDEN_PUBLIC_MARKERS:
        if marker in serialized:
            raise ValueError(f"Exported config still contains private marker {marker!r}")
    config_path.write_text(serialized, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, help="DMD2 iteration directory containing model DCP")
    parser.add_argument("--config-file", type=Path, help="Resolved training run config.yaml")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Student-only output directory")
    parser.add_argument(
        "--sanitize-only",
        action="store_true",
        help="Sanitize and verify an existing framework export without re-exporting weights",
    )
    parser.add_argument("--no-verify", action="store_false", dest="verify", help="Skip the inference smoke test")
    parser.set_defaults(verify=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.sanitize_only:
        if args.checkpoint_path is None or args.config_file is None:
            raise SystemExit("--checkpoint-path and --config-file are required unless --sanitize-only is used")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cosmos_framework.scripts.export_model",
                "--checkpoint-path",
                str(args.checkpoint_path),
                "--config-file",
                str(args.config_file),
                "--student-only-checkpoint-metadata",
                "--no-vit",
                "-o",
                str(args.output_dir),
            ],
            check=True,
        )

    sanitize_student_export_config(args.output_dir)
    if args.verify:
        from cosmos_framework.scripts.export_model import _verify_exported_checkpoint

        _verify_exported_checkpoint(args.output_dir, run_reasoner_check=False)


if __name__ == "__main__":
    main()
