# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for declarative KV cache plan resolution and validation."""

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_plan import (
    KVCacheGroupPlan,
    KVCacheGroupPlanEntry,
    KVCachePlanProvider,
    KVCachePoolPlan,
    KVCachePoolRegion,
    get_profiling_kv_cache_config,
    resolve_kv_cache_plan_provider,
)
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_config_from_groups,
    materialize_kv_cache_group_plan,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

pytestmark = pytest.mark.cpu_test

CUSTOM_PATH = "tests.v1.core.test_kv_cache_plan.CustomProvider"
PLATFORM_PATH = "tests.v1.core.test_kv_cache_plan.PlatformProvider"


class CustomProvider(KVCachePlanProvider):
    group_calls: list[Any] = []
    pool_calls: list[Any] = []

    def get_group_plan(self, vllm_config, kv_cache_specs):
        self.group_calls.append((vllm_config, kv_cache_specs))
        return None

    def get_pool_plan(self, vllm_config, kv_cache_groups):
        self.pool_calls.append((vllm_config, kv_cache_groups))
        return None


class PlatformProvider(KVCachePlanProvider):
    pass


def _config(provider_path=None):
    config = MagicMock()
    config.model_config.kv_cache_plan_provider_cls = provider_path
    return config


def test_resolver_priority_and_sequential_config_isolation():
    default_config = _config()
    model_config = _config(CUSTOM_PATH)

    class TestPlatform(Platform):
        @classmethod
        def get_kv_cache_plan_provider_cls(cls, vllm_config):
            return PLATFORM_PATH if vllm_config is model_config else None

    with patch("vllm.platforms.current_platform", TestPlatform):
        assert resolve_kv_cache_plan_provider(default_config) is None
        assert isinstance(
            resolve_kv_cache_plan_provider(model_config), PlatformProvider
        )
    with (
        patch("vllm.platforms.current_platform", TestPlatform),
        patch.object(TestPlatform, "get_kv_cache_plan_provider_cls", return_value=None),
    ):
        first = resolve_kv_cache_plan_provider(model_config)
        second = resolve_kv_cache_plan_provider(model_config)
        assert isinstance(first, CustomProvider)
        assert isinstance(second, CustomProvider)
        assert first is not second
        assert resolve_kv_cache_plan_provider(default_config) is None


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
    mismatched = KVCachePoolPlan((KVCachePoolRegion(8, ("layer",)),))
    with pytest.raises(ValueError, match="must match"):
        _get_kv_cache_bytes_per_block([group], mismatched)

    spec.page_size_bytes = 8
    sibling = MagicMock(page_size_bytes=8)
    same_group = KVCacheGroupSpec(["layer", "sibling"], spec)
    same_region = KVCachePoolPlan((KVCachePoolRegion(8, ("layer", "sibling")),))
    with (
        patch(
            "vllm.v1.core.kv_cache_utils._get_per_layer_spec",
            side_effect=[spec, sibling],
        ),
        pytest.raises(ValueError, match="one group"),
    ):
        _get_kv_cache_bytes_per_block([same_group], same_region)


def test_pool_plan_rejects_non_block_compact_layout():
    spec = MagicMock(page_size_bytes=512)
    group = KVCacheGroupSpec(["layer"], spec)
    plan = KVCachePoolPlan((KVCachePoolRegion(512, ("layer",)),))
    config = MagicMock()
    config.attention_config.hisparse_config = None
    config.cache_config.get_resolved_kv_cache_layout.return_value = KVCacheLayout.LHBNC

    with pytest.raises(ValueError, match="block-compact"):
        get_kv_cache_config_from_groups(config, [group], 1536, plan)


def test_profiling_uses_resolved_provider_and_restores_override():
    CustomProvider.group_calls.clear()
    CustomProvider.pool_calls.clear()
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
    assert CustomProvider.pool_calls[-1] == (config, ())


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
