# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KV cache config builder resolution and dispatch."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    build_kv_cache_configs,
    build_profiling_kv_cache_config,
    resolve_builder,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

pytestmark = pytest.mark.cpu_test

CUSTOM_PATH = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
PLATFORM_PATH = "tests.v1.core.test_kv_cache_config_builder.PlatformBuilder"


class CustomBuilder(KVCacheConfigBuilder):
    pass


class PlatformBuilder(KVCacheConfigBuilder):
    pass


class ProfilingBuilder(KVCacheConfigBuilder):
    result = MagicMock()
    calls: list[tuple[object, object, object]] = []

    def build_kv_cache_configs(self, vllm_config, kv_cache_specs, available_memory):
        self.calls.append((vllm_config, kv_cache_specs, available_memory))
        return [self.result]

    def build_profiling_kv_cache_config(self, vllm_config, kv_cache_spec, min_blocks):
        self.calls.append((vllm_config, kv_cache_spec, min_blocks))
        return self.result


def _config(builder_cls_path=None):
    config = MagicMock()
    config.model_config.kv_cache_config_builder_cls = builder_cls_path
    return config


def test_resolver_priority_and_sequential_config_isolation():
    default_config = _config()
    model_config = _config(CUSTOM_PATH)

    class TestPlatform(Platform):
        @classmethod
        def get_kv_cache_config_builder_cls(cls, vllm_config):
            if vllm_config is model_config:
                return PLATFORM_PATH
            return None

    with patch("vllm.platforms.current_platform", TestPlatform):
        assert isinstance(resolve_builder(default_config), KVCacheConfigBuilder)
        assert type(resolve_builder(default_config)) is KVCacheConfigBuilder
        assert isinstance(resolve_builder(model_config), PlatformBuilder)

    with (
        patch("vllm.platforms.current_platform", TestPlatform),
        patch.object(
            TestPlatform, "get_kv_cache_config_builder_cls", return_value=None
        ),
    ):
        assert isinstance(resolve_builder(model_config), CustomBuilder)
        assert type(resolve_builder(default_config)) is KVCacheConfigBuilder


def test_profiling_dispatches_to_resolved_builder():
    path = "tests.v1.core.test_kv_cache_config_builder.ProfilingBuilder"
    config = _config(path)
    spec = {"layer": MagicMock()}
    with patch(
        "vllm.platforms.current_platform.get_kv_cache_config_builder_cls",
        return_value=None,
    ):
        result = build_profiling_kv_cache_config(config, spec, 7)
        configs = build_kv_cache_configs(config, [spec], [123])
        legacy_configs = get_kv_cache_configs(config, [spec], [456])
    assert result is ProfilingBuilder.result
    assert ProfilingBuilder.calls[-3] == (config, spec, 7)
    assert configs[0] is ProfilingBuilder.result
    assert legacy_configs[0] is ProfilingBuilder.result
    assert ProfilingBuilder.calls[-2:] == [
        (config, [spec], [123]),
        (config, [spec], [456]),
    ]


def test_glm5_models_declare_builder_in_registry():
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")
    from vllm.model_executor.models.registry import _ModelInfo
    from vllm.models.glm5next import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    expected = "vllm.models.glm5next.kv_cache_config.Glm5NextKVCacheConfigBuilder"
    for model_cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert model_cls.kv_cache_config_builder_cls == expected
        model_info = _ModelInfo.from_model_cls(model_cls)
        assert model_info.kv_cache_config_builder_cls == expected


@pytest.mark.parametrize(
    ("disable_hybrid", "hisparse"), [(True, None), (False, object())]
)
def test_glm5_builder_preserves_generic_grouping_precedence(disable_hybrid, hisparse):
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")
    from vllm.models.glm5next.kv_cache_config import (
        Glm5NextKVCacheConfigBuilder,
    )

    config = _config()
    config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hybrid
    config.attention_config.hisparse_config = hisparse
    builder = Glm5NextKVCacheConfigBuilder()
    with patch.object(
        KVCacheConfigBuilder, "get_kv_cache_groups", return_value=[]
    ) as generic_grouping:
        assert builder.get_kv_cache_groups(config, {}) == []
    generic_grouping.assert_called_once_with(config, {})
