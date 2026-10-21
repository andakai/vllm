# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import (
    get_attn_backend_list_based_on_platform,
    single_gpu_only,
)
from vllm import SamplingParams
from vllm.platforms import current_platform

from .utils import _run_eagle_correctness


@single_gpu_only
def test_qwen3_eagle3_padded_chunk_inputs(monkeypatch, vllm_runner):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    prompt = list(range(1000, 1192))
    chunk_inputs = []
    with vllm_runner(
        "Qwen/Qwen3-8B",
        enforce_eager=True,
        max_model_len=512,
        max_num_batched_tokens=64,
        max_num_seqs=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config={
            "method": "eagle3",
            "model": "AngelSlim/Qwen3-8B_eagle3",
            "num_speculative_tokens": 1,
        },
    ) as runner:
        core = runner.llm.llm_engine.engine_core.engine_core
        drafter = core.model_executor.driver_worker.worker.model_runner.drafter

        def observe_inputs(module, args, kwargs):
            positions = kwargs["positions"].tolist()
            if len(positions) > 1 and max(positions) < len(prompt) - 1:
                chunk_inputs.append((positions, kwargs["input_ids"].tolist()))

        handle = drafter.model.register_forward_pre_hook(
            observe_inputs, with_kwargs=True
        )
        try:
            runner.llm.generate(
                [{"prompt_token_ids": prompt}],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
            )
        finally:
            handle.remove()
    assert len(chunk_inputs) == 2
    for positions, token_ids in chunk_inputs:
        assert token_ids == [prompt[position + 1] for position in positions]


@single_gpu_only
@pytest.mark.skipif(
    current_platform.is_device_capability_family(100),
    reason="DeepSeek head_dim=192 not supported on SM100/SM110 (Blackwell)",
)
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            (
                "eagle",
                "eagle618/deepseek-v3-random",
                "eagle618/eagle-deepseek-v3-random",
                1,
            ),
            False,
            False,
            "auto",
            0.0,
        ),
    ],
    ids=["deepseek_eagle"],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_light(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )


@single_gpu_only
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "transformers",
            0.8,
            # TODO(hmellor): figure out why memory usage is so high
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(),
                reason="Feature is experimental and uses too much memory in CUDA CI",
            ),
        ),
        (
            (
                "eagle3",
                "Qwen/Qwen3-VL-8B-Instruct",
                "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3",
                1,
            ),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Rayzl/qwen2.5-vl-7b-eagle3-sgl",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
            # TODO: Re-measure the reference threshold when re-enabling this case.
            # Text-only mode starts but loses target correctness with ROCm/TRITON;
            # full multimodal profiling currently fails during engine startup.
            marks=pytest.mark.skip(
                reason="Qwen2.5-VL Eagle3 is not yet reliable in this test setup"
            ),
        ),
        (
            (
                "eagle3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
        ),
    ],
    ids=[
        "qwen3_eagle3",
        "qwen3_eagle3-transformers",
        "qwen3_vl_eagle3",
        "qwen2_5_vl_eagle3",
        "llama3_eagle3",
    ],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_medium(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )
