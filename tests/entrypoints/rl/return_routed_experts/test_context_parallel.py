# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import numpy as np
import pybase64 as base64

from tests.utils import RemoteOpenAIServer, multi_gpu_test

# MLA, so DCP works at TP=2. 27 layers (layer 0 is dense), 64 experts, top-6.
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPT = list(range(1000, 2000))
MAX_TOKENS = 8


def generate(client) -> tuple[np.ndarray, int]:
    result = client.completions.create(
        model=MODEL_NAME, prompt=PROMPT, max_tokens=MAX_TOKENS, temperature=0
    )
    encoded = result.model_dump()["choices"][0]["routed_experts"]
    rows = np.load(io.BytesIO(base64.b64decode(encoded)))
    details = result.usage.prompt_tokens_details
    return rows, details.cached_tokens if details else 0


@multi_gpu_test(num_gpus=2)
def test_routed_experts_with_decode_context_parallel():
    """With DCP, rows are stored per DCP-scaled block; a repeated prompt must get
    the stored rows back for the blocks it hits in the prefix cache."""
    args = [
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--tensor-parallel-size",
        "2",
        "--decode-context-parallel-size",
        "2",
        "--enable-return-routed-experts",
        "--enable-prompt-tokens-details",
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as server:
        client = server.get_client()
        cold, _ = generate(client)
        warm, cached = generate(client)

    assert cold.shape == (len(PROMPT) + MAX_TOKENS - 1, 27, 6)
    assert cached > 0
    np.testing.assert_array_equal(warm[:cached], cold[:cached])
