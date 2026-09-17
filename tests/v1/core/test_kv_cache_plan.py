# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for composable KV cache planning middleware."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_plan import (
    KVCachePlanningRequest,
    KVCachePoolRegion,
    build_kv_cache_configs,
    build_profiling_kv_cache_config,
    resolve_kv_cache_planner,
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

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

PLANNER_PATH = "tests.v1.core.test_kv_cache_plan.RecordingPlanner"


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


class RecordingPlanner:
    instances: list["RecordingPlanner"] = []
    calls: list[KVCachePlanningRequest] = []

    def __init__(self, delegate):
        self.delegate = delegate
        self.instances.append(self)

    def __call__(self, request):
        self.calls.append(request)
        return [_special_config()]


class WrappedPlanner:
    def __init__(self, delegate):
        self.delegate = delegate

    def __call__(self, request):
        return self.delegate(request)


def _config(*, planner_path=None):
    config = MagicMock()
    config.model_config.kv_cache_planner_cls = planner_path
    return config


def test_resolver_creates_fresh_model_planner_per_config():
    RecordingPlanner.instances.clear()
    config = _config(planner_path=PLANNER_PATH)

    first = resolve_kv_cache_planner(config)
    second = resolve_kv_cache_planner(config)

    assert isinstance(first, RecordingPlanner)
    assert isinstance(second, RecordingPlanner)
    assert first is not second


@pytest.mark.parametrize("mode", ["delegate", "wrap", "takeover"])
def test_platform_composes_model_aware_delegate(mode):
    config = _config(planner_path=PLANNER_PATH)

    class TestPlatform(Platform):
        seen_delegate = None

        @classmethod
        def get_kv_cache_planner(cls, vllm_config, delegate):
            cls.seen_delegate = delegate
            if mode == "delegate":
                return delegate
            if mode == "wrap":
                return WrappedPlanner(delegate)
            return lambda request: [_special_config()]

    with patch("vllm.platforms.current_platform", TestPlatform):
        resolved = resolve_kv_cache_planner(config)

    assert isinstance(TestPlatform.seen_delegate, RecordingPlanner)
    if mode == "delegate":
        assert resolved is TestPlatform.seen_delegate
    elif mode == "wrap":
        assert resolved.delegate is TestPlatform.seen_delegate
    else:
        assert resolved is not TestPlatform.seen_delegate


def test_platform_callbacks_take_precedence_over_model_callbacks():
    platform_group_planner = MagicMock()
    platform_region_planner = MagicMock()
    model_group_planner = MagicMock()
    model_region_planner = MagicMock()
    captured = []

    def core(request):
        captured.append(request)
        return []

    class ModelPlanner:
        def __init__(self, delegate):
            self.delegate = delegate

        def __call__(self, request):
            return self.delegate(
                request.with_declarative_plan(model_group_planner, model_region_planner)
            )

    model = ModelPlanner(core)
    request = KVCachePlanningRequest(
        MagicMock(),
        [],
        [],
        group_planner=platform_group_planner,
        region_planner=platform_region_planner,
    )

    model(request)

    assert captured[0].group_planner is platform_group_planner
    assert captured[0].region_planner is platform_region_planner


def test_same_planner_entry_handles_final_and_profiling():
    RecordingPlanner.calls.clear()
    config = _config(planner_path=PLANNER_PATH)
    spec = {"layer": MagicMock()}

    final = build_kv_cache_configs(config, [spec], [123])
    profiling = build_profiling_kv_cache_config(config, spec, 7)

    assert final[0].num_blocks == profiling.num_blocks == 23
    assert [request.fixed_num_blocks for request in RecordingPlanner.calls] == [None, 7]


def test_public_helper_and_platform_takeover_allow_arbitrary_geometry():
    config = _config()

    class TestPlatform(Platform):
        @classmethod
        def get_kv_cache_planner(cls, vllm_config, delegate):
            return lambda request: [_special_config()]

    with patch("vllm.platforms.current_platform", TestPlatform):
        result = get_kv_cache_configs(config, [{"layer": MagicMock()}], [1])[0]

    assert result.num_blocks == 23
    assert result.kv_cache_tensors == [
        KVCacheTensor(4096, ["primary", "noncontiguous"], 301, 73, offset=11),
        KVCacheTensor(4096, ["alias"], 509, 91, offset=312),
    ]


def _default_config():
    return SimpleNamespace(
        num_prefill_lookahead_tokens=0,
        model_config=SimpleNamespace(
            original_max_model_len=16,
            max_model_len=16,
            kv_cache_planner_cls=None,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=41,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
    )


def test_fixed_num_blocks_does_not_mutate_config_override():
    config = _default_config()
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )

    result = build_kv_cache_configs(config, [{"layer": spec}], [0], fixed_num_blocks=3)

    assert result[0].num_blocks == 3
    assert config.cache_config.num_gpu_blocks_override == 41


def test_core_normalizes_specs_before_declarative_callbacks():
    seen_specs = None

    def group_planner(vllm_config, kv_cache_specs):
        nonlocal seen_specs
        seen_specs = dict(kv_cache_specs)
        return None

    class DeclarativePlanner:
        def __init__(self, delegate):
            self.delegate = delegate

        def __call__(self, request):
            return self.delegate(
                request.with_declarative_plan(
                    group_planner, lambda config, groups: None
                )
            )

    config = _default_config()
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    full = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    sliding = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=8,
    )

    DeclarativePlanner(resolve_kv_cache_planner(config))(
        KVCachePlanningRequest(config, [{"full": full, "sliding": sliding}], [0], 3)
    )

    assert seen_specs is not None
    assert all(
        isinstance(spec, FullAttentionSpec) and not isinstance(spec, SlidingWindowSpec)
        for spec in seen_specs.values()
    )


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

    with pytest.raises(ValueError, match="must match"):
        _get_kv_cache_bytes_per_block([group], (KVCachePoolRegion(4, ("layer",)),))


def test_pool_plan_rejects_non_block_compact_layout():
    spec = MagicMock(page_size_bytes=512)
    group = KVCacheGroupSpec(["layer"], spec)
    regions = (KVCachePoolRegion(512, ("layer",)),)
    config = MagicMock()
    config.attention_config.hisparse_config = None
    config.cache_config.get_resolved_kv_cache_layout.return_value = KVCacheLayout.LHBNC

    with pytest.raises(ValueError, match="block-compact"):
        get_kv_cache_config_from_groups(config, [group], 1536, regions)


def test_glm5_middleware_preserves_shared_regions():
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")
    from vllm.models.glm5next.kv_cache_plan import Glm5NextKVCachePlanner

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
            block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16
        ),
        "indexer": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.uint8,
            tokens_per_state=2,
        ),
    }
    planner = Glm5NextKVCachePlanner(lambda request: [])
    groups = planner._get_kv_cache_groups(config, specs)
    assert groups is not None
    regions = planner._get_kv_cache_regions(config, tuple(groups))

    assert regions is not None
    assert regions[0].layers == ("mla", "mamba.0", "mamba.1", "mamba.2")
    assert regions[1].layers == ("indexer",)


def test_glm5_models_declare_middleware_in_registry():
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")
    from vllm.model_executor.models.registry import _ModelInfo
    from vllm.models.glm5next import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    expected = "vllm.models.glm5next.kv_cache_plan.Glm5NextKVCachePlanner"
    for model_cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert model_cls.kv_cache_planner_cls == expected
        assert _ModelInfo.from_model_cls(model_cls).kv_cache_planner_cls == expected
