# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from cosmos_predict2._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder, get_text_embedding


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def encoder(device):
    return CosmosT5TextEncoder(device=device)


@pytest.mark.L2
def test_single_prompt(encoder, device):
    prompt = "This is a test prompt."
    embedding = get_text_embedding(prompt, encoder=encoder, device=device)
    assert embedding.shape == (1, 512, 1024)


@pytest.mark.L2
def test_multiple_prompts(encoder, device):
    prompts = ["First prompt.", "Second prompt.", "Third prompt."]
    embeddings = get_text_embedding(prompts, encoder=encoder, device=device)
    assert embeddings.shape == (3, 512, 1024)


@pytest.mark.L2
def test_global_encoder(device):
    prompt = "Testing global encoder."
    embedding1 = get_text_embedding(prompt, device=device)
    embedding2 = get_text_embedding(prompt, device=device)
    assert torch.allclose(embedding1, embedding2)


@pytest.mark.L2
def test_custom_max_length(encoder, device):
    prompt = "Short prompt."
    max_length = 20
    embedding = get_text_embedding(prompt, encoder=encoder, device=device, max_length=max_length)
    assert embedding.shape == (1, max_length, 1024)


@pytest.mark.L2
def test_encoder_device(encoder):
    assert encoder.device in ["cuda", "cpu"]
    assert next(encoder.text_encoder.parameters()).device.type == encoder.device


@pytest.mark.L2
def test_empty_prompt_list(encoder, device):
    with pytest.raises(ValueError, match="The input prompt list is empty."):
        get_text_embedding([], encoder=encoder, device=device)


@pytest.mark.L2
def test_long_prompt(encoder, device):
    long_prompt = "This is a very long prompt. " * 100
    embedding = get_text_embedding(long_prompt, encoder=encoder, device=device)
    assert embedding.shape == (1, 512, 1024)  # Should be truncated to max_length
