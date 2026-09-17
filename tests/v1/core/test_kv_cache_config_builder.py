# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    resolve_kv_cache_config_builder,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

pytestmark = pytest.mark.cpu_test

_MODULE = "tests.v1.core.test_kv_cache_config_builder"


class ModelBuilder(KVCacheConfigBuilder):
    pass


class OtherModelBuilder(KVCacheConfigBuilder):
    pass


class PlatformBuilder(KVCacheConfigBuilder):
    pass


class BuilderPlatform(Platform):
    @classmethod
    def get_kv_cache_config_builder_cls(cls, vllm_config):
        return f"{_MODULE}.PlatformBuilder"


def _config(builder: str | None = None):
    return SimpleNamespace(
        model_config=SimpleNamespace(kv_cache_config_builder_cls=builder)
    )


def test_resolver_priority_and_sequential_config_isolation():
    first = _config(f"{_MODULE}.ModelBuilder")
    second = _config(f"{_MODULE}.OtherModelBuilder")
    with patch("vllm.platforms.current_platform", Platform):
        assert isinstance(resolve_kv_cache_config_builder(first), ModelBuilder)
        assert isinstance(resolve_kv_cache_config_builder(second), OtherModelBuilder)
        assert type(resolve_kv_cache_config_builder(_config())) is KVCacheConfigBuilder

    with patch("vllm.platforms.current_platform", BuilderPlatform):
        assert isinstance(resolve_kv_cache_config_builder(first), PlatformBuilder)


def test_profiling_config_uses_resolved_hooks_and_restores_override():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=17,
            prefix_cache_retention_interval=None,
        )
    )
    builder = KVCacheConfigBuilder()
    expected = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])
    with (
        patch.object(builder, "get_kv_cache_groups", return_value=[]) as groups,
        patch.object(
            builder, "get_kv_cache_config_from_groups", return_value=expected
        ) as build,
        patch(
            "vllm.v1.kv_cache_spec_registry."
            "KVCacheSpecRegistry.check_kv_cache_spec_registry"
        ),
    ):
        assert builder.get_profiling_kv_cache_config(config, {}, 3) is expected
    groups.assert_called_once_with(config, {})
    build.assert_called_once_with(config, [], available_memory=0)
    assert config.cache_config.num_gpu_blocks_override == 17


@pytest.mark.parametrize(
    ("module_name", "attributes"),
    [
        (
            "vllm.v1.worker.gpu.cudagraph_utils",
            ("_init_minimal_kv_cache_for_profiling",),
        ),
        (
            "vllm.v1.worker.gpu_model_runner",
            ("GPUModelRunner", "_init_minimal_kv_cache_for_profiling"),
        ),
    ],
)
def test_worker_profiling_routes_through_resolved_builder(module_name, attributes):
    builder = MagicMock(spec=KVCacheConfigBuilder)
    minimal = KVCacheConfig(num_blocks=5, kv_cache_tensors=[], kv_cache_groups=[])
    builder.get_profiling_kv_cache_config.return_value = minimal
    runner = SimpleNamespace(
        vllm_config=object(),
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        max_num_reqs=8,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=5),
        get_kv_cache_spec=MagicMock(return_value={}),
        initialize_kv_cache=MagicMock(),
    )
    profiling_fn: Any = __import__(module_name, fromlist=[attributes[0]])
    for attribute in attributes:
        profiling_fn = getattr(profiling_fn, attribute)
    with patch(
        "vllm.v1.core.kv_cache_config_builder.resolve_kv_cache_config_builder",
        return_value=builder,
    ):
        profiling_fn(runner)
    builder.get_profiling_kv_cache_config.assert_called_once_with(
        runner.vllm_config, {}, 5
    )
    runner.initialize_kv_cache.assert_called_once_with(minimal, is_profiling=True)
    assert runner.cache_config.num_gpu_blocks == 5
