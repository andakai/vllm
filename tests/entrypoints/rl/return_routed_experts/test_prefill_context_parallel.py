# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io

import numpy as np
import pybase64 as base64

from tests.utils import RemoteOpenAIServer, multi_gpu_test

# MLA, 27 layers (layer 0 is dense), 64 experts, top-6.
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPT = list(range(1000, 3000))
MAX_TOKENS = 8
ARGS = [
    "--max-model-len",
    "4096",
    "--enforce-eager",
    "--tensor-parallel-size",
    "2",
    "--enable-return-routed-experts",
    "--enable-prompt-tokens-details",
]


def generate(client) -> tuple[np.ndarray, int]:
    result = client.completions.create(
        model=MODEL_NAME, prompt=PROMPT, max_tokens=MAX_TOKENS, temperature=0
    )
    encoded = result.model_dump()["choices"][0]["routed_experts"]
    rows = np.load(io.BytesIO(base64.b64decode(encoded)))
    details = result.usage.prompt_tokens_details
    return rows, details.cached_tokens if details else 0


@multi_gpu_test(num_gpus=4)
def test_routed_experts_with_prefill_context_parallel():
    """With PCP, each rank routes part of the prompt; the rows must come back in
    token order. The first MoE layer's input differs from TP alone only by
    attention rounding, so its rows match a server without PCP; misordered
    rows would not. Prefix-cache hits get the stored rows back."""
    with RemoteOpenAIServer(
        MODEL_NAME, [*ARGS, "--prefill-context-parallel-size", "2"]
    ) as server:
        client = server.get_client()
        cold, _ = generate(client)
        warm, cached = generate(client)
    with RemoteOpenAIServer(MODEL_NAME, ARGS) as server:
        reference, _ = generate(server.get_client())

    assert cold.shape == (len(PROMPT) + MAX_TOKENS - 1, 27, 6)
    prompt_rows = len(PROMPT) - 1
    same = (cold[:prompt_rows, 1] == reference[:prompt_rows, 1]).all(axis=-1)
    assert same.mean() > 0.99, f"first MoE layer matches at {same.mean():.4f}"
    assert cached > 0
    np.testing.assert_array_equal(warm[:cached], cold[:cached])
