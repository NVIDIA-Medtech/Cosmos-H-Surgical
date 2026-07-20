# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

RunModule = Callable[..., dict[str, Any]]


def register_experiments() -> ModuleType:
    """Import project-owned experiment registrations before framework composition."""
    from cosmos_h_surgical.configs import cosmos3_sft

    return cosmos3_sft


def run_framework_cli(
    argv: Sequence[str],
    *,
    run_module: RunModule = runpy.run_module,
    register: Callable[[], ModuleType] = register_experiments,
) -> int:
    """Run the pinned training entrypoint after registering surgical experiments."""
    os.environ["COSMOS_TRAINING"] = "1"
    register()

    original_argv = sys.argv
    sys.argv = ["cosmos-h-surgical train", *argv]
    try:
        run_module("cosmos_framework.scripts.train", run_name="__main__")
    except ModuleNotFoundError as error:
        if error.name and error.name != "cosmos_framework":
            raise RuntimeError(
                "Cosmos training dependencies are not installed. "
                "Run `uv sync --frozen --extra train` before post-training."
            ) from error
        raise
    finally:
        sys.argv = original_argv
    return 0
