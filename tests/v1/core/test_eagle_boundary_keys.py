# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE prefix-cache hits keep their last block once the cache writer is known
to have read the same next token (``CacheConfig.eagle_boundary_tokens``)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_boundary_keys,
    get_request_block_hasher,
    init_none_hash,
    resolve_eagle_boundary_tokens,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK = 16
FULL = FullAttentionSpec(
    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
)
SWA = SlidingWindowSpec(
    block_size=BLOCK,
    num_kv_heads=1,
    head_size=1,
    dtype=torch.float32,
    sliding_window=3 * BLOCK,
)
MAMBA = MambaSpec(
    block_size=BLOCK,
    shapes=((1, 1),),
    dtypes=(torch.float32,),
    mamba_cache_mode="align",
)


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _config(*specs: KVCacheSpec, eagle_groups: tuple[int, ...] = ()) -> KVCacheConfig:
    groups = [KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)]
    for i in eagle_groups:
        groups[i].is_eagle_group = True
    return KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        eagle_boundary_tokens=1,
    )


def _manager(config: KVCacheConfig) -> KVCacheManager:
    return KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
        use_eagle=True,
    )


def _request(request_id: str, tokens: list[int], **kwargs) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
        **kwargs,
    )


def _prefill(manager: KVCacheManager, request: Request) -> None:
    blocks, num_hit, _ = manager.get_computed_blocks(request)
    manager.allocate_slots(request, request.num_tokens - num_hit, num_hit, blocks)


def _hit(manager: KVCacheManager, tokens: list[int]) -> int:
    return manager.get_computed_blocks(_request("probe", tokens))[1]


@pytest.mark.parametrize("hybrid", [False, True])
def test_hit_keeps_last_block_only_when_next_token_matches(hybrid: bool):
    # Hybrid: only the full-attention group holds draft layers.
    config = _config(FULL, SWA, eagle_groups=(0,)) if hybrid else _config(FULL)
    manager = _manager(config)
    prompt = list(range(6 * BLOCK + 1))
    writer = _request("writer", prompt)
    _prefill(manager, writer)
    manager.free(writer)

    # The legacy drop would stop at 5 blocks.
    assert _hit(manager, prompt) == 6 * BLOCK
    assert _hit(manager, prompt[:-1] + [999]) == 5 * BLOCK


def test_boundary_keys_recorded_when_next_token_is_appended():
    manager = _manager(_config(FULL))
    prompt = list(range(2 * BLOCK))
    writer = _request("writer", prompt)
    _prefill(manager, writer)

    # The prompt's last block waits for the token sampled after it.
    assert _hit(manager, prompt + [7]) == BLOCK
    writer.append_output_token_ids(7)
    manager.record_boundary_keys(writer)
    assert _hit(manager, prompt + [7]) == 2 * BLOCK
    assert _hit(manager, prompt + [8]) == BLOCK


def test_hit_picks_cached_copy_with_matching_boundary_keys():
    manager = _manager(_config(FULL))
    prompt = list(range(2 * BLOCK))
    tails = {}
    for next_token in (7, 8):
        # No prefix lookup, so each writer caches its own copy.
        writer = _request(str(next_token), prompt + [next_token])
        manager.allocate_slots(writer, writer.num_tokens)
        tails[next_token] = manager.coordinator.get_blocks(writer.request_id)[0][1]

    for next_token, tail in tails.items():
        probe = _request("probe", prompt + [next_token])
        blocks, num_hit, _ = manager.get_computed_blocks(probe)
        assert num_hit == 2 * BLOCK
        assert blocks.blocks[0][1] is tail


def test_boundary_keys_identify_multimodal_inputs():
    feature = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=BLOCK, length=4),
        identifier="image",
        modality="image",
    )
    request = _request("mm", [5] * (2 * BLOCK), mm_features=[feature])

    assert get_boundary_keys(request, BLOCK - 1, 2) == (5, ("image", 0))
    assert get_boundary_keys(request, 2 * BLOCK, 1) is None


def _vllm_config(
    method: str = "eagle3",
    lookahead: int = 1,
    connector: bool | None = None,
    disabled: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            enable_mamba_shared_prefix_checkpoint=False,
        ),
        speculative_config=SimpleNamespace(
            method=method,
            disable_eagle_boundary_check=disabled,
            use_eagle_block_drop=lambda: True,
        ),
        num_prefill_lookahead_tokens=lookahead,
        attention_config=SimpleNamespace(hisparse_config=None),
        # ``connector`` is whether the connector supports boundary keys.
        kv_transfer_config=None if connector is None else connector,
    )


@pytest.mark.parametrize(
    ("config", "vllm_kwargs", "expected"),
    [
        (_config(FULL), {}, 1),
        (_config(FULL), {"method": "dflash"}, 0),
        (_config(FULL), {"lookahead": 2}, 0),  # Multi-module MTP.
        (_config(FULL), {"disabled": True}, 0),
        (_config(FULL), {"connector": False}, 1),
        # Unflagged groups all count as draft groups, SWA cannot verify.
        (_config(FULL, SWA), {}, 0),
        (_config(FULL, SWA, eagle_groups=(0,)), {}, 1),
        # Mamba never drops, but connector tiers must place checkpoints alike.
        (_config(FULL, MAMBA), {}, 1),
        (_config(FULL, MAMBA), {"connector": True}, 1),
        (_config(FULL, MAMBA), {"connector": False}, 0),
    ],
)
def test_resolve_eagle_boundary_tokens(monkeypatch, config, vllm_kwargs, expected):
    monkeypatch.setattr(
        KVConnectorFactory,
        "get_connector_class",
        lambda supported: SimpleNamespace(
            supports_eagle_boundary_keys=lambda _: supported
        ),
    )
    vllm_config = _vllm_config(**vllm_kwargs)
    assert resolve_eagle_boundary_tokens(config, vllm_config) == expected
