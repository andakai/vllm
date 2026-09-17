# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for declarative KV cache plan resolution and validation."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_plan import (
    KVCacheGroupPlan,
    KVCacheGroupPlanEntry,
    KVCachePlanProvider,
    KVCachePoolRegion,
    KVCacheWorkerPlan,
    RegionWorkerPlan,
    get_profiling_kv_cache_config,
    resolve_kv_cache_plan_provider,
)
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    materialize_kv_cache_group_plan,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

pytestmark = pytest.mark.cpu_test

CUSTOM_PATH = "tests.v1.core.test_kv_cache_plan.CustomProvider"
PLATFORM_PATH = "tests.v1.core.test_kv_cache_plan.PlatformProvider"


class CustomProvider(KVCachePlanProvider):
    group_calls: list[Any] = []
    worker_calls: list[Any] = []

    def get_group_plan(self, vllm_config, kv_cache_specs):
        self.group_calls.append((vllm_config, kv_cache_specs))
        return None

    def get_worker_plan(self, vllm_config, kv_cache_groups):
        self.worker_calls.append((vllm_config, kv_cache_groups))
        return super().get_worker_plan(vllm_config, kv_cache_groups)


class PlatformProvider(KVCachePlanProvider):
    pass


class SyntheticWorkerPlan(KVCacheWorkerPlan):
    def __init__(self, groups, materialized):
        self.groups = groups
        self.materialized = materialized

    @property
    def bytes_per_block(self):
        return 256

    def max_memory_usage_bytes(self, vllm_config):
        return (vllm_config.model_config.max_model_len // 4 + 3) * 256

    def materialize(self, vllm_config, num_blocks):
        self.materialized.append(num_blocks)
        size = self.bytes_per_block * num_blocks
        tensors = [
            KVCacheTensor(
                size=size,
                layers=[name],
                layer_stride=17,
                block_stride=256,
                offset=0 if name != "b" else 96,
            )
            for group in self.groups
            for name in group.layer_names
        ]
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=list(self.groups),
        )


class SyntheticProvider(KVCachePlanProvider):
    def __init__(self):
        super().__init__()
        self.materialized: list[list[int]] = []

    def get_group_plan(self, vllm_config, kv_cache_specs):
        return KVCacheGroupPlan(
            (KVCacheGroupPlanEntry(tuple(kv_cache_specs.items()), True),)
        )

    def get_worker_plan(self, vllm_config, kv_cache_groups):
        calls: list[int] = []
        self.materialized.append(calls)
        return SyntheticWorkerPlan(kv_cache_groups, calls)


def _synthetic_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=8,
            original_max_model_len=None,
            kv_cache_plan_provider_cls=None,
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        num_prefill_lookahead_tokens=0,
    )


def _glm_config(pp_size=1):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=16,
            original_max_model_len=None,
            kv_cache_plan_provider_cls=None,
            get_total_num_hidden_layers=lambda: 8,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            mamba_cache_mode="none",
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            decode_context_parallel_size=1,
        ),
        num_prefill_lookahead_tokens=0,
    )


def _glm_specs():
    specs = {}
    for index in range(8):
        if index % 4 == 3:
            specs[f"layers.{index}.attn"] = MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=64,
                dtype=torch.bfloat16,
            )
            specs[f"layers.{index}.indexer"] = MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.uint8,
                tokens_per_state=4,
            )
        else:
            specs[f"layers.{index}.linear_attn"] = MambaSpec(
                block_size=16,
                shapes=((128,),),
                dtypes=(torch.float32,),
                num_speculative_blocks=2,
            )
    return specs


def _config(provider_path=None):
    config = MagicMock()
    config.model_config.kv_cache_plan_provider_cls = provider_path
    config.num_prefill_lookahead_tokens = 0
    config.attention_config.hisparse_config = None
    return config


def test_resolver_priority_and_sequential_config_isolation():
    default_config = _config()
    model_config = _config(CUSTOM_PATH)

    class TestPlatform(Platform):
        @classmethod
        def get_kv_cache_plan_provider_cls(cls, vllm_config):
            return PLATFORM_PATH if vllm_config is model_config else None

    with patch("vllm.platforms.current_platform", TestPlatform):
        default_provider = resolve_kv_cache_plan_provider(default_config)
        assert type(default_provider) is KVCachePlanProvider
        platform_provider = resolve_kv_cache_plan_provider(model_config)
        assert isinstance(platform_provider, PlatformProvider)
        assert isinstance(platform_provider._delegate, CustomProvider)
    with (
        patch("vllm.platforms.current_platform", TestPlatform),
        patch.object(TestPlatform, "get_kv_cache_plan_provider_cls", return_value=None),
    ):
        first = resolve_kv_cache_plan_provider(model_config)
        second = resolve_kv_cache_plan_provider(model_config)
        assert isinstance(first, CustomProvider)
        assert isinstance(second, CustomProvider)
        assert first is not second
        default_provider = resolve_kv_cache_plan_provider(default_config)
        assert type(default_provider) is KVCachePlanProvider


def test_plan_data_is_frozen_and_core_validates_it():
    spec = MagicMock()
    entry = KVCacheGroupPlanEntry((("layer", spec),))
    plan = KVCacheGroupPlan((entry,))
    with pytest.raises(FrozenInstanceError):
        plan.groups = ()
    assert materialize_kv_cache_group_plan({"layer": spec}, plan)[0].layer_names == [
        "layer"
    ]

    duplicate = KVCacheGroupPlan((entry, entry))
    with pytest.raises(ValueError, match="more than once"):
        materialize_kv_cache_group_plan({"layer": spec}, duplicate)

    group = KVCacheGroupSpec(["layer"], spec)
    spec.page_size_bytes = 4
    with pytest.raises(ValueError, match="must match"):
        RegionWorkerPlan((group,), (KVCachePoolRegion(8, ("layer",)),))

    spec.page_size_bytes = 8
    sibling = MagicMock(page_size_bytes=8)
    same_group = KVCacheGroupSpec(["layer", "sibling"], spec)
    with (
        patch(
            "vllm.v1.core.kv_cache_plan._get_layer_specs",
            return_value={"layer": spec, "sibling": sibling},
        ),
        pytest.raises(ValueError, match="one group"),
    ):
        RegionWorkerPlan((same_group,), (KVCachePoolRegion(8, ("layer", "sibling")),))


def test_pool_plan_rejects_non_block_compact_layout():
    spec = MagicMock(page_size_bytes=512)
    group = KVCacheGroupSpec(["layer"], spec)
    plan = RegionWorkerPlan((group,), (KVCachePoolRegion(512, ("layer",)),))
    config = MagicMock()
    config.attention_config.hisparse_config = None
    config.cache_config.get_resolved_kv_cache_layout.return_value = KVCacheLayout.LHBNC

    with pytest.raises(ValueError, match="block-compact"):
        get_kv_cache_config_from_groups(config, [group], 1536, plan)


def test_custom_worker_plan_owns_capacity_placement_and_rank_rematerialization():
    config = _synthetic_config()
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
    )
    specs = {name: spec for name in ("a", "b", "c")}
    provider = SyntheticProvider()

    configs = get_kv_cache_configs(
        config,
        [specs, specs],
        [8 * 256, 6 * 256],
        plan_provider=provider,
    )

    assert [cache.num_blocks for cache in configs] == [6, 6]
    assert provider.materialized == [[8, 6], [6]]
    assert {
        tensor.layers[0]: (tensor.offset, tensor.layer_stride, tensor.block_stride)
        for tensor in configs[0].kv_cache_tensors
    } == {
        "a": (0, 17, 256),
        "b": (96, 17, 256),
        "c": (0, 17, 256),
    }

    with pytest.raises(ValueError, match="max seq len"):
        get_kv_cache_configs(
            config,
            [specs],
            [5 * 256],
            plan_provider=SyntheticProvider(),
        )


def test_core_normalizes_specs_before_custom_grouping():
    config = _synthetic_config()
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    full = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
    )
    sliding = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=8,
    )

    cache = get_kv_cache_configs(
        config,
        [{"full": full, "sliding": sliding}],
        [6 * 256],
        plan_provider=SyntheticProvider(),
    )[0]

    group_spec = cache.kv_cache_groups[0].kv_cache_spec
    assert isinstance(group_spec, UniformTypeKVCacheSpecs)
    assert all(
        isinstance(spec, FullAttentionSpec) and not isinstance(spec, SlidingWindowSpec)
        for spec in group_spec.kv_cache_specs.values()
    )


def test_profiling_runs_the_same_custom_worker_plan_pipeline():
    config = _synthetic_config()
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
    )
    provider = SyntheticProvider()
    with patch(
        "vllm.v1.core.kv_cache_utils.resolve_kv_cache_plan_provider",
        return_value=provider,
    ):
        cache = get_profiling_kv_cache_config(config, {"a": spec}, 3)

    assert cache.num_blocks == 3
    assert cache.kv_cache_tensors[0].block_stride == 256
    assert provider.materialized == [[3]]


def test_glm_region_plan_pp_profiling_and_rank_alignment():
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")
    from vllm.models.glm5next.kv_cache_plan import Glm5NextKVCachePlanProvider

    specs = _glm_specs()
    worker_specs = [
        {name: spec for name, spec in specs.items() if int(name.split(".")[1]) < 4},
        {name: spec for name, spec in specs.items() if int(name.split(".")[1]) >= 4},
    ]
    provider = Glm5NextKVCachePlanProvider()
    pp_config = _glm_config(pp_size=2)
    main_page = specs["layers.3.attn"].page_size_bytes
    index_page = specs["layers.3.indexer"].page_size_bytes
    bytes_per_block = main_page + index_page

    configs = get_kv_cache_configs(
        pp_config,
        worker_specs,
        [40 * bytes_per_block, 30 * bytes_per_block],
        plan_provider=provider,
    )

    assert [config.num_blocks for config in configs] == [30, 30]
    for rank, config in enumerate(configs):
        first_layer = rank * 4
        tensors = {
            layer: tensor
            for tensor in config.kv_cache_tensors
            for layer in tensor.layers
        }
        attn = tensors[f"layers.{first_layer + 3}.attn"]
        for index in range(first_layer, first_layer + 3):
            assert tensors[f"layers.{index}.linear_attn"].offset == attn.offset
        assert {tensor.size for tensor in config.kv_cache_tensors} == {
            30 * bytes_per_block
        }

    profiling_config = _glm_config()
    with patch(
        "vllm.v1.core.kv_cache_utils.resolve_kv_cache_plan_provider",
        return_value=Glm5NextKVCachePlanProvider(),
    ):
        profiling = get_profiling_kv_cache_config(profiling_config, specs, 4)
    assert profiling.num_blocks == 4
    assert {tensor.size for tensor in profiling.kv_cache_tensors} == {
        2 * bytes_per_block * 4
    }


def test_profiling_uses_resolved_provider_and_restores_override():
    CustomProvider.group_calls.clear()
    CustomProvider.worker_calls.clear()
    config = _config(CUSTOM_PATH)
    config.cache_config.num_gpu_blocks_override = 17
    config.cache_config.prefix_cache_retention_interval = None
    with patch(
        "vllm.platforms.current_platform.get_kv_cache_plan_provider_cls",
        return_value=None,
    ):
        result = get_profiling_kv_cache_config(config, {}, 3)
    assert result.num_blocks == 1
    assert config.cache_config.num_gpu_blocks_override == 17
    assert CustomProvider.group_calls[-1][0] is config
    assert CustomProvider.worker_calls[-1] == (config, ())


def test_glm5_models_declare_provider_in_registry():
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")
    from vllm.model_executor.models.registry import _ModelInfo
    from vllm.models.glm5next import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    expected = "vllm.models.glm5next.kv_cache_plan.Glm5NextKVCachePlanProvider"
    for model_cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert model_cls.kv_cache_plan_provider_cls == expected
        model_info = _ModelInfo.from_model_cls(model_cls)
        assert model_info.kv_cache_plan_provider_cls == expected
