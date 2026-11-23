# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from tests.utils import RemoteOpenAIServer, multi_gpu_test

from .test_openai_api import MODEL_NAME, decode_routed_experts

PROMPTS = ["Hello, world", "The capital of France is", "1 2 3 4 5 6 7 8"]


def routed_experts(pipeline_parallel_size: int) -> list[np.ndarray]:
    args = [
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--enable-return-routed-experts",
        "--hf-overrides",
        '{"sliding_window": null}',
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
    ]
    with RemoteOpenAIServer(MODEL_NAME, args) as server:
        client = server.get_client()
        return [
            decode_routed_experts(
                client.completions.create(
                    model=MODEL_NAME, prompt=prompt, max_tokens=10, temperature=0
                ).model_dump()["choices"][0]["routed_experts"]
            )
            for prompt in PROMPTS
        ]


@multi_gpu_test(num_gpus=2)
def test_routed_experts_match_without_pipeline_parallelism():
    """With PP=2, each stage owns one of the two layers; the rows the last stage
    returns must include the first stage's layer."""
    for pp_rows, rows in zip(routed_experts(2), routed_experts(1), strict=True):
        np.testing.assert_array_equal(pp_rows, rows)
