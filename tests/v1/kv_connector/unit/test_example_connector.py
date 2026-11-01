# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import NamedTuple

import pytest
import torch
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import AttentionConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.forward_context import ForwardContext
from vllm.inputs import TextPrompt
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from .utils import create_vllm_config, set_mtp_config

MODEL_NAME = "RedHatAI/Qwen2.5-VL-3B-Instruct-quantized.w8a8"

SAMPLING_PARAMS = SamplingParams(temperature=0.0, top_k=1, max_tokens=128)

TEXT_PROMPTS = [
    "What's in the image(s)? Around 30 words. What's special in 2nd image?",
    "The future of AI is",
]


def test_lookahead_eagle3_chunked_prefill(vllm_runner, tmp_path):
    """Real target/draft KV survives external reload and rejects a changed suffix."""
    with vllm_runner(
        "Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        dtype="bfloat16",
        seed=42,
        block_size=16,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        kv_cache_memory_bytes=1 << 30,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_log_stats=False,
        attention_config={"backend": "FLASH_ATTN"},
        speculative_config={
            "method": "eagle3",
            "model": "AngelSlim/Qwen3-1.7B_eagle3",
            "revision": "94441b48acc5804677ae12259617c83323b543a9",
            "num_speculative_tokens": 4,
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
    ) as runner:
        llm = runner.get_llm()
        assert llm.llm_engine.vllm_config.num_hash_lookahead_tokens == 1
        tokenizer = llm.get_tokenizer()
        tokens = tokenizer.encode(
            "The library lends twelve books every Monday. " * 40,
            add_special_tokens=False,
        )[:136]
        sampling = SamplingParams(temperature=0, seed=42, max_tokens=64, logprobs=5)

        def counter(name):
            return next(
                metric.value for metric in llm.get_metrics() if metric.name == name
            )

        def generate(prompt):
            # ExampleConnector finishes its synchronous save before request completion.
            assert llm.reset_prefix_cache(reset_connector=False)
            before = counter("vllm:external_prefix_cache_hits")
            result = llm.generate(
                [{"prompt_token_ids": prompt}], sampling, use_tqdm=False
            )[0].outputs[0]
            return result, counter("vllm:external_prefix_cache_hits") - before

        cold, cold_hits = generate(tokens)
        warm, warm_hits = generate(tokens)
        assert cold_hits == 0
        assert warm_hits == 128
        assert counter("vllm:spec_decode_num_accepted_tokens") > 0
        assert cold.logprobs is not None and warm.logprobs is not None
        for token in (cold.token_ids[0], warm.token_ids[0]):
            assert math.exp(cold.logprobs[0][token].logprob) == pytest.approx(
                math.exp(warm.logprobs[0][token].logprob), abs=0.01, rel=0
            )
        changed = tokens.copy()
        changed[128] = tokenizer.encode("elephant", add_special_tokens=False)[0]
        assert changed[128] != tokens[128]
        _, changed_hits = generate(changed)
        assert changed_hits == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("hash_block_size", [8, 16])
@pytest.mark.parametrize("lookahead", [1, 17])
def test_lookahead_atomic_store_and_restore(tmp_path, hash_block_size, lookahead):
    """Publish target/draft KV together, keyed by the complete lookahead window."""
    config = create_vllm_config(
        kv_connector="ExampleConnector",
        kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
    )
    set_mtp_config(config, lookahead)
    # Exercise the resolved H contract beyond today's one-token MTP window.
    config.num_hash_lookahead_tokens = lookahead
    cache = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target", "draft"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float32
                ),
            )
        ],
        hash_block_size=hash_block_size,
    )
    scheduler = ExampleConnector(config, KVConnectorRole.SCHEDULER, cache)
    worker = ExampleConnector(config, KVConnectorRole.WORKER, cache)
    init_none_hash(sha256)
    hasher = get_request_block_hasher(hash_block_size, sha256, lookahead)

    def request(tokens, name="test"):
        return Request(
            request_id=name,
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=4),
            pooling_params=None,
            block_hasher=hasher,
        )

    def step(req, ids, scheduled):
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [NewRequestData.from_request(req, (ids,))]
        output.num_scheduled_tokens = {req.request_id: scheduled}
        output.total_num_scheduled_tokens = scheduled
        return output

    tokens = list(range(32 + lookahead))
    req = request(tokens)
    blocks = KVCacheBlocks(([KVCacheBlock(1)],))
    scheduler.update_state_after_alloc(req, blocks, 0)
    assert scheduler.build_connector_meta(step(req, [1], 16)).requests == []
    req.num_computed_tokens = 16
    continuation = step(req, [], 16)
    continuation.scheduled_new_reqs = []
    continuation.scheduled_cached_reqs.req_ids = [req.request_id]
    continuation.scheduled_cached_reqs.new_block_ids = [([2],)]
    meta = scheduler.build_connector_meta(continuation)
    assert len(meta.requests) == 1 and len(meta.requests[0].slot_mapping) == 32
    assert scheduler.get_num_new_matched_tokens(req, 0) == (0, False)

    layers = {
        name: torch.arange(8 * 16 * 8, dtype=torch.float32).reshape(8, 1, 16, 8)
        for name in ("target", "draft")
    }
    worker.register_kv_caches(layers)
    worker.bind_connector_metadata(meta)
    # Draft KV is finalized after metadata creation, before wait_for_save.
    layers["draft"].add_(1000)
    expected = {name: layer[1:3].clone() for name, layer in layers.items()}
    worker.wait_for_save()
    worker.clear_connector_metadata()
    assert len(list(tmp_path.iterdir())) == 1
    assert scheduler.get_num_new_matched_tokens(req, 0) == (32, False)
    assert scheduler.get_num_new_matched_tokens(req, 16) == (16, False)
    changed = tokens.copy()
    changed[32] += 100
    assert scheduler.get_num_new_matched_tokens(request(changed), 0) == (0, False)

    hit = request(tokens, "hit")
    load_blocks = KVCacheBlocks(([KVCacheBlock(i) for i in (4, 5, 6, 7)],))
    scheduler.update_state_after_alloc(hit, load_blocks, 32)
    load_meta = scheduler.build_connector_meta(step(hit, [4, 5, 6, 7], lookahead))
    assert len(load_meta.requests) == 1 and not load_meta.requests[0].is_store
    for layer in layers.values():
        layer.zero_()
    worker.bind_connector_metadata(load_meta)
    worker.start_load_kv(
        ForwardContext(no_compile_layers={}, attn_metadata=None, slot_mapping={})
    )
    worker.clear_connector_metadata()
    for name, layer in layers.items():
        torch.testing.assert_close(layer[4:6], expected[name], rtol=0, atol=0)
        assert torch.count_nonzero(layer[:4]) == 0
        assert torch.count_nonzero(layer[6:]) == 0


class InputCase(NamedTuple):
    text: str
    img: list[Image]
    expected_len: int
    info: str


def _check_path_len(path):
    """Return the latest length in path."""
    return len(list(path.iterdir()))


def _list_path(path):
    """Return the list of foldername (hashes generated) under the path."""
    return list(path.iterdir())


def run_test(
    tmp_path,
    processor,
    llm: LLM,
    question: str,
    image_urls: list[Image],
    expected_len: int,
    info: str,
):
    """One individual test to process the prompt and output base on 1 set of input
    Then check if the length in the storage path matches the expected length
    `info` introduces details or purpose of the individual test
    """
    print(f"***info: {info}***")
    print(f"**Expected storage path length after llm generate: {expected_len}**")
    process_prompt(processor, llm, question, image_urls)

    print(f"Path matched expected length: {_check_path_len(tmp_path)}")
    print(f"Hashes under the storage path: {_list_path(tmp_path)}")

    assert _check_path_len(tmp_path) == expected_len, (
        f"Expect storage path length {expected_len} ;",
        f"but end up {_check_path_len(tmp_path)} instead. ",
        f"Info: {info}",
    )


def process_prompt(processor, llm: LLM, question: str, image_urls: list[Image]):
    """Form the prompt based on the text and image input, then llm generate output."""
    placeholders = [
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(image_pil)},
        }
        for image_pil in image_urls
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_input: TextPrompt = {"prompt": prompt}
    if image_urls:
        prompt_input["multi_modal_data"] = {"image": image_urls}

    outputs = llm.generate(prompt_input, sampling_params=SAMPLING_PARAMS)

    print("-" * 50)
    print("Output:")
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)


@pytest.mark.parametrize(
    "attn_backend",
    (
        ["FLASH_ATTN", "TRITON_ATTN"]
        if current_platform.is_cuda()
        else ["TRITON_ATTN"]
        if current_platform.is_rocm() or current_platform.is_xpu()
        else []
    ),
)
def test_shared_storage_connector_hashes(tmp_path, attn_backend):
    """Tests that ExampleConnector saves KV to the storage locations
    with proper hashes; that are unique for inputs with identical text but
    different images (same size), or same multiple images but different orders.
    """
    # Using tmp_path as the storage path to store KV
    print(f"KV storage path at: {str(tmp_path)}")

    # don't put this import at the top level
    # it will call torch.accelerator.device_count()
    from transformers import AutoProcessor

    # Create processor to handle the chat prompt
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Prepare images for the tests
    # Resize to the same size to check hashes correctness
    image_1 = ImageAsset("stop_sign").pil_image.resize((1280, 720))
    image_2 = ImageAsset("cherry_blossom").pil_image.resize((1280, 720))

    # Make sure that they are not the same picture
    assert image_1 != image_2, "The images should not be identical"

    # Create the LLM instance
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=8192,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        attention_config=AttentionConfig(backend=attn_backend),
        enforce_eager=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
        ),
        limit_mm_per_prompt={"image": 2},
    )

    # Prepare the input cases
    input_cases = [
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=1,
            info="image_1 single input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the first time. "
                "It is in same pixel size with image_1, yet it "
                "should be able to form a new unique hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1],
            expected_len=2,
            info=(
                "image_1 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2],
            expected_len=2,
            info=(
                "image_2 single input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=3,
            info="image_1 with image_2 input the first time.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info="The image order is swapped. Should form new hash.",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_1, image_2],
            expected_len=4,
            info=(
                "[image_1, image_2] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[image_2, image_1],
            expected_len=4,
            info=(
                "[image_2, image_1] input the 2nd time. "
                "It should not form another new hash."
            ),
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Pure text input test as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[0],
            img=[],
            expected_len=5,
            info="Identical pure text input as a case-control",
        ),
        InputCase(
            text=TEXT_PROMPTS[1],
            img=[],
            expected_len=6,
            info="Another pure text input as a case-control",
        ),
    ]

    # Run tests
    for case_id, (text, img, expected_len, info) in enumerate(input_cases):
        print("\n", "=" * 25, f"Below running input case: {case_id}", "=" * 25)
        run_test(tmp_path, processor, llm, text, img, expected_len, info)

    print("All tests passed successfully!")
