# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from typing import Any, cast

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.renderers.chat_encode_cache import ChatEncodeCache
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import ChatParams, TokenizeParams
from vllm.tokenizers import get_tokenizer

# Byte-level BPE (ChatML) and SentencePiece-style Metaspace tokenizers.
MODELS = ["Qwen/Qwen3-0.6B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]


def _content(rng: random.Random, tokens: list[str], n: int) -> str:
    # Whitespace, partial words and added-token text inside message content.
    pieces = [" tool", "call", "\n", "  ", "unbeliev", "able", "é", "中文", "\r\n"]
    pieces += [f" {t}\n" for t in tokens[:8]]
    return "".join(rng.choice(pieces) for _ in range(n))


@pytest.mark.parametrize("model", MODELS)
def test_growing_conversation_matches_full_encode(model: str) -> None:
    tokenizer = cast(Any, get_tokenizer(model))
    cache = ChatEncodeCache.maybe_create(tokenizer, max_bytes=1 << 22)
    assert cache is not None
    added = [t.content for t in tokenizer.added_tokens_decoder.values()]
    rng = random.Random(0)

    messages = [{"role": "system", "content": _content(rng, added, 50)}]
    previous = ""
    for _ in range(6):
        messages.append({"role": "user", "content": _content(rng, added, 200)})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Only text after the previous prompt (and its trailing generation
        # prompt) needs encoding.
        encoded_chars = sum(
            len(s) for s in set(cache.split(prompt)) if s not in cache._segments
        )
        assert encoded_chars <= len(prompt) - len(previous) + 64
        assert (
            cache.encode(prompt)
            == tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        messages.append({"role": "assistant", "content": _content(rng, added, 200)})
        previous = prompt


def test_split_cuts_only_before_added_tokens() -> None:
    cache = ChatEncodeCache.maybe_create(get_tokenizer(MODELS[0]), max_bytes=1 << 22)
    assert cache is not None
    text = "unbelievable<|im_end|>\n<|im_start|>user\nhi"
    assert cache.split(text) == ["unbelievable", "<|im_end|>\n", "<|im_start|>user\nhi"]


def test_cache_is_bounded() -> None:
    cache = ChatEncodeCache.maybe_create(get_tokenizer(MODELS[0]), max_bytes=200)
    assert cache is not None
    for i in range(20):
        cache.encode(f"<|im_start|>user\nmessage number {i}")
    assert 0 < cache._num_bytes <= 200


@pytest.mark.asyncio
async def test_renderer_token_ids_unchanged() -> None:
    renderers = [
        HfRenderer(
            VllmConfig(
                model_config=ModelConfig(
                    MODELS[0], enable_incremental_chat_tokenization=enabled
                )
            ),
            get_tokenizer(MODELS[0]),
        )
        for enabled in (False, True)
    ]
    try:
        assert renderers[1]._chat_encode_cache is not None
        messages = []
        for turn in range(3):
            messages.append({"role": "user", "content": f"Question {turn}?"})
            sync = [
                cast(dict, r.render_messages(messages, ChatParams())[1])
                for r in renderers
            ]
            async_ = [
                cast(dict, (await r.render_messages_async(messages, ChatParams()))[1])
                for r in renderers
            ]
            expected = sync[0]["prompt_token_ids"]
            for prompt in sync[1:] + async_:
                assert prompt["prompt_token_ids"] == expected
            messages.append({"role": "assistant", "content": f"Answer {turn}."})
    finally:
        for renderer in renderers:
            renderer.shutdown()


def test_text_prompt_tokenization_uses_cache() -> None:
    """The online server renders chat prompts to text and encodes them with
    tokenize_prompts; that path returns the same ids through the cache."""
    renderers = [
        HfRenderer(
            VllmConfig(
                model_config=ModelConfig(
                    MODELS[0], enable_incremental_chat_tokenization=enabled
                )
            ),
            get_tokenizer(MODELS[0]),
        )
        for enabled in (False, True)
    ]
    try:
        tokenizer = renderers[0].get_tokenizer()
        messages = [
            {"role": "user", "content": "Question one?"},
            {"role": "assistant", "content": "Answer one."},
            {"role": "user", "content": "Question two?"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = TokenizeParams(max_total_tokens=4096, add_special_tokens=False)
        ids = [
            r.tokenize_prompts([{"prompt": text}], params)[0]["prompt_token_ids"]
            for r in renderers
        ]
        assert ids[1] == ids[0]
        cache = renderers[1]._chat_encode_cache
        assert cache is not None and len(cache._segments) > 1
    finally:
        for renderer in renderers:
            renderer.shutdown()
