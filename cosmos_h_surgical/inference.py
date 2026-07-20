# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from typing import Any


def run_framework_cli(
    argv: Sequence[str],
    *,
    entrypoint: Callable[[], Any] | None = None,
) -> int:
    """Run the pinned Cosmos Framework inference CLI with unchanged arguments."""
    os.environ["COSMOS_TRAINING"] = "0"
    if entrypoint is None:
        from cosmos_framework.scripts.inference import main as entrypoint

    original_argv = sys.argv
    sys.argv = ["cosmos-h-surgical infer", *argv]
    try:
        result = entrypoint()
    finally:
        sys.argv = original_argv
    return 0 if result is None else int(result)
