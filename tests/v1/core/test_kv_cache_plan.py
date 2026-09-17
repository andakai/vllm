# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KV cache builder composition and declarative plans."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    build_kv_cache_configs,
    build_profiling_kv_cache_config,
    resolve_kv_cache_config_builder,
)
from vllm.v1.core.kv_cache_plan import (
    KVCachePlanProvider,
    KVCachePoolRegion,
)
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    validate_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

pytestmark = pytest.mark.cpu_test

PROVIDER_PATH = "tests.v1.core.test_kv_cache_plan.CustomProvider"
BUILDER_PATH = "tests.v1.core.test_kv_cache_plan.RecordingBuilder"


class CustomProvider(KVCachePlanProvider):
    pass


def _special_config() -> KVCacheConfig:
    spec = MagicMock(page_size_bytes=128)
    return KVCacheConfig(
        num_blocks=23,
        kv_cache_groups=[KVCacheGroupSpec(["primary", "noncontiguous", "alias"], spec)],
        kv_cache_tensors=[
            KVCacheTensor(4096, ["primary", "noncontiguous"], 301, 73, offset=11),
            KVCacheTensor(4096, ["alias"], 509, 91, offset=312),
        ],
    )


class RecordingBuilder(KVCacheConfigBuilder):
    calls: list[tuple[object, object, object, int | None]] = []

    def build(
        self,
        vllm_config,
        kv_cache_specs,
        available_memory,
        *,
        fixed_num_blocks=None,
    ):
        self.calls.append(
            (vllm_config, kv_cache_specs, available_memory, fixed_num_blocks)
        )
        return [_special_config()]


class WrappedBuilder(KVCacheConfigBuilder):
    def __init__(self, delegate):
        self.delegate = delegate

    def build(self, *args, **kwargs):
        return self.delegate.build(*args, **kwargs)


class TakeoverBuilder(KVCacheConfigBuilder):
    def build(self, *args, **kwargs):
        return [_special_config()]


def _config(*, provider_path=None, builder_path=None):
    config = MagicMock()
    config.model_config.kv_cache_plan_provider_cls = provider_path
    config.model_config.kv_cache_config_builder_cls = builder_path
    return config


def test_fresh_model_aware_builder_per_config():
    first_config = _config(provider_path=PROVIDER_PATH)
    second_config = _config()

    first = resolve_kv_cache_config_builder(first_config)
    again = resolve_kv_cache_config_builder(first_config)
    second = resolve_kv_cache_config_builder(second_config)

    assert isinstance(first.plan_provider, CustomProvider)
    assert isinstance(again.plan_provider, CustomProvider)
    assert first is not again
    assert first.plan_provider is not again.plan_provider
    assert second.plan_provider is None


@pytest.mark.parametrize("mode", ["delegate", "wrap", "takeover"])
def test_platform_composes_model_aware_delegate(mode):
    config = _config(builder_path=BUILDER_PATH)

    class TestPlatform(Platform):
        seen_delegate = None

        @classmethod
        def get_kv_cache_config_builder(cls, vllm_config, delegate):
            cls.seen_delegate = delegate
            if mode == "delegate":
                return delegate
            if mode == "wrap":
                return WrappedBuilder(delegate)
            return TakeoverBuilder()

    with patch("vllm.platforms.current_platform", TestPlatform):
        resolved = resolve_kv_cache_config_builder(config)

    assert isinstance(TestPlatform.seen_delegate, RecordingBuilder)
    if mode == "delegate":
        assert resolved is TestPlatform.seen_delegate
    elif mode == "wrap":
        assert resolved.delegate is TestPlatform.seen_delegate
    else:
        assert isinstance(resolved, TakeoverBuilder)


def test_same_full_builder_entry_handles_final_and_profiling():
    RecordingBuilder.calls.clear()
    config = _config(builder_path=BUILDER_PATH)
    spec = {"layer": MagicMock()}

    final = build_kv_cache_configs(config, [spec], [123])
    profiling = build_profiling_kv_cache_config(config, spec, 7)

    assert final[0].num_blocks == profiling.num_blocks == 23
    assert [call[3] for call in RecordingBuilder.calls] == [None, 7]


def test_public_planning_helper_uses_resolved_builder():
    config = _config(builder_path=BUILDER_PATH)

    result = get_kv_cache_configs(config, [{"layer": MagicMock()}], [1])

    assert result[0].num_blocks == 23


def test_fixed_num_blocks_does_not_mutate_config_override():
    config = SimpleNamespace(
        num_prefill_lookahead_tokens=0,
        model_config=SimpleNamespace(
            original_max_model_len=16,
            max_model_len=16,
            kv_cache_plan_provider_cls=None,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=41,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
    )
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )

    result = KVCacheConfigBuilder().build(
        config, [{"layer": spec}], [0], fixed_num_blocks=3
    )

    assert result[0].num_blocks == 3
    assert config.cache_config.num_gpu_blocks_override == 41


def test_core_normalizes_specs_before_declarative_provider():
    class CapturingProvider(KVCachePlanProvider):
        seen_specs = None

        def get_kv_cache_groups(self, vllm_config, kv_cache_specs):
            self.seen_specs = dict(kv_cache_specs)
            return None

    config = SimpleNamespace(
        num_prefill_lookahead_tokens=0,
        model_config=SimpleNamespace(
            original_max_model_len=16,
            max_model_len=16,
            kv_cache_plan_provider_cls=None,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
    )
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

    provider = CapturingProvider()
    KVCacheConfigBuilder(provider).build(
        config,
        [{"full": full, "sliding": sliding}],
        [0],
        fixed_num_blocks=3,
    )

    assert provider.seen_specs is not None
    assert all(
        isinstance(spec, FullAttentionSpec) and not isinstance(spec, SlidingWindowSpec)
        for spec in provider.seen_specs.values()
    )


def test_full_builder_can_return_unconstrained_tensor_geometry():
    config = _config(builder_path=BUILDER_PATH)

    class TestPlatform(Platform):
        @classmethod
        def get_kv_cache_config_builder(cls, vllm_config, delegate):
            return TakeoverBuilder()

    with patch("vllm.platforms.current_platform", TestPlatform):
        result = build_kv_cache_configs(config, [{"layer": MagicMock()}], [1])[0]

    assert result.num_blocks == 23
    assert result.kv_cache_tensors == [
        KVCacheTensor(4096, ["primary", "noncontiguous"], 301, 73, offset=11),
        KVCacheTensor(4096, ["alias"], 509, 91, offset=312),
    ]


def test_declarative_regions_are_frozen_and_core_validated():
    spec = MagicMock(page_size_bytes=8)
    group = KVCacheGroupSpec(["layer"], spec)
    region = KVCachePoolRegion(8, ("layer",))
    with pytest.raises(FrozenInstanceError):
        region.layers = ()
    validate_kv_cache_groups({"layer": spec}, [group])
    assert _get_kv_cache_bytes_per_block([group], (region,)) == 8

    with pytest.raises(ValueError, match="more than once"):
        validate_kv_cache_groups({"layer": spec}, [group, group])

    mismatched = (KVCachePoolRegion(4, ("layer",)),)
    with pytest.raises(ValueError, match="must match"):
        _get_kv_cache_bytes_per_block([group], mismatched)

    sibling = MagicMock(page_size_bytes=8)
    same_group = KVCacheGroupSpec(["layer", "sibling"], spec)
    same_region = (KVCachePoolRegion(8, ("layer", "sibling")),)
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
    regions = (KVCachePoolRegion(512, ("layer",)),)
    config = MagicMock()
    config.attention_config.hisparse_config = None
    config.cache_config.get_resolved_kv_cache_layout.return_value = KVCacheLayout.LHBNC

    with pytest.raises(ValueError, match="block-compact"):
        get_kv_cache_config_from_groups(config, [group], 1536, regions)


def test_glm5_declarative_adapter_preserves_shared_regions():
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")
    from vllm.models.glm5next.kv_cache_plan import Glm5NextKVCachePlanProvider

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    mamba = MambaSpec(
        block_size=16,
        shapes=((2, 8), (3, 2, 2)),
        dtypes=(torch.float16, torch.float16),
        mamba_cache_mode="none",
        num_speculative_blocks=0,
    )
    specs = {
        **{f"mamba.{index}": mamba for index in range(3)},
        "mla": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=32,
            dtype=torch.float16,
        ),
        "indexer": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.uint8,
            tokens_per_state=2,
        ),
    }
    provider = Glm5NextKVCachePlanProvider()

    groups = provider.get_kv_cache_groups(config, specs)
    assert groups is not None
    regions = provider.get_kv_cache_regions(config, tuple(groups))

    assert regions is not None
    assert regions[0].layers == ("mla", "mamba.0", "mamba.1", "mamba.2")
    assert regions[1].layers == ("indexer",)


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
        assert model_info.kv_cache_config_builder_cls is None
