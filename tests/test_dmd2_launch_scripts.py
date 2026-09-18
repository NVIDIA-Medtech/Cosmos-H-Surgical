# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_cosmos_h_surgical_transfer_dmd2.sh"


def test_rank_zero_initializes_shared_run_directory_before_other_nodes() -> None:
    script = RUNNER.read_text()
    handshake_start = script.index('launch_ready="${run_dir}/.launch_ready_${DMD2_RDZV_ID}"')
    handshake_end = script.index("if (( DMD2_NNODES == 1 )); then", handshake_start)
    handshake = script[handshake_start:handshake_end]

    rank_zero = handshake.index('if [[ "${DMD2_NODE_RANK}" == "0" ]]; then')
    fresh_guard = handshake.index('if [[ -z "${DMD2_RESUME_CHECKPOINT}"', rank_zero)
    create_logs = handshake.index('mkdir -p "${run_dir}/logs"', fresh_guard)
    nonzero_branch = handshake.index("else", create_logs)
    wait_for_rank_zero = handshake.index('[[ -s "${launch_ready}" ]] && break', nonzero_branch)

    assert rank_zero < fresh_guard < create_logs < nonzero_branch < wait_for_rank_zero
    assert 'mkdir -p "${run_dir}/logs"' not in handshake[nonzero_branch:]
