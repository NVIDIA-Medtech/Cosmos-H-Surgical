# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Register the public Predict and Transfer post-training experiments."""

from cosmos_h_surgical.configs import predict_sft, transfer_sft

__all__ = ["predict_sft", "transfer_sft"]
