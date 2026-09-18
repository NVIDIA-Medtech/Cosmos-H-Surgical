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
CompatInstaller = Callable[[], None]


def _install_distributed_affinity_compat(distributed: ModuleType | None = None) -> None:
    """Constrain framework NVML affinity to the current Slurm cpuset."""
    if distributed is None:
        from cosmos_framework.utils import distributed as framework_distributed

        distributed = framework_distributed

    marker = "_cosmos_h_surgical_cpuset_affinity_compat"
    if getattr(distributed, marker, False):
        return

    original_device = distributed.Device

    def cpuset_aware_device(*args: Any, **kwargs: Any) -> Any:
        device = original_device(*args, **kwargs)
        original_get_cpu_affinity = device.get_cpu_affinity

        def get_cpu_affinity() -> list[int]:
            requested = set(original_get_cpu_affinity())
            try:
                allowed = set(os.sched_getaffinity(0))
            except (AttributeError, OSError) as error:
                distributed.log.warning(f"Unable to read the current CPU affinity; using NVML affinity: {error}")
                return sorted(requested)

            affinity = requested & allowed
            if affinity:
                return sorted(affinity)

            distributed.log.warning(
                f"GPU-local CPUs {sorted(requested)} do not overlap allowed CPUs {sorted(allowed)}; "
                "keeping the current Slurm cpuset"
            )
            return sorted(allowed)

        device.get_cpu_affinity = get_cpu_affinity
        return device

    distributed.Device = cpuset_aware_device
    setattr(distributed, marker, True)


def register_experiments() -> ModuleType:
    """Import project-owned experiment registrations before framework composition."""
    from cosmos_h_surgical.configs import cosmos3_sft

    return cosmos3_sft


def run_framework_cli(
    argv: Sequence[str],
    *,
    run_module: RunModule = runpy.run_module,
    register: Callable[[], ModuleType] = register_experiments,
    install_compat: CompatInstaller = _install_distributed_affinity_compat,
) -> int:
    """Run the pinned training entrypoint after registering surgical experiments."""
    os.environ["COSMOS_TRAINING"] = "1"
    register()
    install_compat()

    original_argv = sys.argv
    sys.argv = ["cosmos-h-surgical train", *argv]
    try:
        run_module("cosmos_framework.scripts.train", run_name="__main__")
    except ModuleNotFoundError as error:
        if error.name and error.name != "cosmos_framework":
            raise RuntimeError(
                "Cosmos training dependencies are not installed. "
                "Run `uv sync --group cu130` for CUDA 13 or "
                "`uv sync --group cu128` for CUDA 12.8 before post-training."
            ) from error
        raise
    finally:
        sys.argv = original_argv
    return 0
