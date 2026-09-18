# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Register the public post-training and distillation experiments."""

from cosmos_h_surgical.configs import predict_sft, transfer_dmd2, transfer_sft

__all__ = ["predict_sft", "transfer_dmd2", "transfer_sft"]
