# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import check_logprobs_close
from tests.utils import single_gpu_only
from vllm import SamplingParams, TokensPrompt

from ...utils import compute_acceptance_len
from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "google/gemma-4-E4B-it", 1, "google/gemma-4-E4B-it-assistant"),
            False,
            0.50,
        ),  # gemma4 MTP with assistant model, ref: ~62%
    ],
    ids=["gemma4-e4b"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@single_gpu_only
def test_gemma4_mtp_reuses_prefix_before_divergent_token(vllm_runner):
    """Shared target KV remains reusable when the next prompt token changes."""
    sampling = SamplingParams(temperature=0, max_tokens=32, logprobs=5)
    with vllm_runner(
        "google/gemma-4-E4B-it",
        max_model_len=512,
        block_size=16,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "audio": 0},
        speculative_config={
            "method": "mtp",
            "model": "google/gemma-4-E4B-it-assistant",
            "num_speculative_tokens": 2,
        },
    ) as runner:
        llm = runner.llm
        config = llm.llm_engine.vllm_config
        assert config.num_prefill_lookahead_tokens == 1
        assert config.num_hash_lookahead_tokens == 0
        tokenizer = llm.get_tokenizer()
        prefix = tokenizer.encode("Describe the natural world around us. " * 40)[:128]
        yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no = tokenizer.encode("No", add_special_tokens=False)[0]
        assert len(prefix) == 128 and yes != no
        writer = TokensPrompt(prompt_token_ids=[*prefix, yes])
        sibling = TokensPrompt(prompt_token_ids=[*prefix, no])

        cold = llm.generate([sibling], sampling)[0]
        assert llm.reset_prefix_cache()
        llm.generate([writer], sampling)
        before_warm = llm.get_metrics()
        warm = llm.generate([sibling], sampling)[0]

        assert cold.num_cached_tokens == 0
        assert warm.num_cached_tokens == len(prefix)
        assert compute_acceptance_len(llm.get_metrics(), before_warm) > 1
        assert len(cold.outputs[0].token_ids) == len(warm.outputs[0].token_ids) > 0
        check_logprobs_close(
            outputs_0_lst=[
                (
                    list(cold.outputs[0].token_ids),
                    cold.outputs[0].text,
                    cold.outputs[0].logprobs,
                )
            ],
            outputs_1_lst=[
                (
                    list(warm.outputs[0].token_ids),
                    warm.outputs[0].text,
                    warm.outputs[0].logprobs,
                )
            ],
            name_0="cold_gemma4_mtp",
            name_1="warm_gemma4_mtp",
        )
