# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

import vllm.v1.core.kv_cache_config_builder as builder_module
from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    _get_profiling_kv_cache_config,
    get_kv_cache_config_builder,
)
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)


def _make_vllm_config(builder_cls_path: str | None = None) -> MagicMock:
    """Create a minimal mock VllmConfig for builder resolution tests."""
    cfg = MagicMock()
    cfg.model_config.kv_cache_config_builder_cls = builder_cls_path
    return cfg


CUSTOM_PATH = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
DEFAULT_PATH = "vllm.v1.core.kv_cache_planning.DefaultKVCacheConfigBuilder"


class CustomBuilder(DefaultKVCacheConfigBuilder):
    """A test builder subclass."""

    pass


class ExactBlocksBuilder(DefaultKVCacheConfigBuilder):
    """Record the capacity passed through the profiling path."""

    seen_num_blocks: int | None = None

    def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
        return []

    def get_kv_cache_config_from_groups(self, vllm_config, kv_cache_groups, num_blocks):
        type(self).seen_num_blocks = num_blocks
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )


@pytest.fixture(autouse=True)
def _reset_active_builder():
    # Clear the cached builder directly; the cache attribute is private on
    # purpose, so tests poke it instead of shipping a reset() production API.
    builder_module._active_builder = None
    ExactBlocksBuilder.seen_num_blocks = None
    yield
    builder_module._active_builder = None
    ExactBlocksBuilder.seen_num_blocks = None


class TestPlatformHookResolution:
    """The platform hook owns the resolution priority."""

    def test_default_hook_prefers_model_declaration(self):
        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        assert Platform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH

    def test_default_hook_falls_back_to_default_builder(self):
        cfg = _make_vllm_config(builder_cls_path=None)
        assert Platform.get_kv_cache_config_builder_cls(cfg) == DEFAULT_PATH


class TestBuilderResolution:
    @patch("vllm.platforms.current_platform")
    def test_resolves_default_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        assert type(get_kv_cache_config_builder(cfg)) is DefaultKVCacheConfigBuilder

    @patch("vllm.platforms.current_platform")
    def test_model_declared_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert isinstance(get_kv_cache_config_builder(cfg), CustomBuilder)

    @patch("vllm.platforms.current_platform")
    def test_resolves_once_and_caches(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert get_kv_cache_config_builder(cfg) is get_kv_cache_config_builder(cfg)

    @patch("vllm.platforms.current_platform")
    def test_reset_forces_resolution_again(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        first = get_kv_cache_config_builder(cfg)
        builder_module._active_builder = None
        second = get_kv_cache_config_builder(cfg)
        assert first is not second
        assert isinstance(second, CustomBuilder)

    @patch("vllm.platforms.current_platform")
    def test_resolved_builder_uses_instance_entry_point(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        active = get_kv_cache_config_builder(cfg)
        assert isinstance(active, CustomBuilder)
        with patch.object(active, "get_kv_cache_configs", return_value=[]) as g:
            assert active.get_kv_cache_configs(cfg, [], [0]) == []
            g.assert_called_once()

    def test_public_interface_declares_entry_point_and_two_hooks(self):
        assert KVCacheConfigBuilder.__abstractmethods__ == {
            "get_kv_cache_configs",
            "get_kv_cache_groups",
            "get_kv_cache_config_from_groups",
        }
        assert issubclass(DefaultKVCacheConfigBuilder, KVCacheConfigBuilder)

    @patch(
        "vllm.v1.core.kv_cache_planning.KVCacheSpecRegistry."
        "check_kv_cache_spec_registry"
    )
    def test_capacity_is_derived_from_unit_placement(self, _mock_check):
        class PlacementBuilder(DefaultKVCacheConfigBuilder):
            seen_num_blocks: list[int] = []

            def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
                return [KVCacheGroupSpec(["layer"], MagicMock())]

            def get_kv_cache_config_from_groups(
                self, vllm_config, kv_cache_groups, num_blocks
            ):
                self.seen_num_blocks.append(num_blocks)
                return KVCacheConfig(
                    num_blocks=num_blocks,
                    kv_cache_tensors=[
                        KVCacheTensor(
                            size=64 * num_blocks,
                            layers=["layer"],
                            layer_stride=64 * num_blocks,
                            block_stride=64,
                        )
                    ],
                    kv_cache_groups=kv_cache_groups,
                )

            def _get_max_memory_usage_bytes_from_groups(
                self, vllm_config, kv_cache_groups
            ):
                return 0

        cfg = _make_vllm_config()
        cfg.num_prefill_lookahead_tokens = 0
        cfg.attention_config.hisparse_config = None
        cfg.cache_config.num_gpu_blocks_override = None
        cfg.model_config.original_max_model_len = 1
        cfg.model_config.max_model_len = 1
        builder = PlacementBuilder()

        (config,) = builder.get_kv_cache_configs(cfg, [{"layer": MagicMock()}], [223])

        assert builder.seen_num_blocks == [1, 3]
        assert config.num_blocks == 3
        assert {tensor.size for tensor in config.kv_cache_tensors} == {192}

    @patch("vllm.platforms.current_platform")
    def test_profiling_reuses_exact_block_materializer(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = (
            "tests.v1.core.test_kv_cache_config_builder.ExactBlocksBuilder"
        )
        cfg = _make_vllm_config()
        cfg.cache_config.num_gpu_blocks_override = 11

        result = _get_profiling_kv_cache_config(cfg, {}, min_blocks=7)

        assert result.num_blocks == 7
        assert ExactBlocksBuilder.seen_num_blocks == 7
        assert cfg.cache_config.num_gpu_blocks_override == 11


class TestPlatformCustomPriority:
    """A vendor platform can override the hook to customize priority."""

    def test_platform_builder_wins_over_model_declaration(self):
        class PlatformFirstPlatform(Platform):
            @classmethod
            def get_kv_cache_config_builder_cls(cls, vllm_config):
                return CUSTOM_PATH

        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        # Model declares CustomBuilder too; the platform forces it anyway.
        assert PlatformFirstPlatform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH
        with patch("vllm.platforms.current_platform", PlatformFirstPlatform):
            assert isinstance(get_kv_cache_config_builder(cfg), CustomBuilder)

    def test_platform_delegates_to_model_declaration(self):
        class ModelFirstPlatform(Platform):
            @classmethod
            def get_kv_cache_config_builder_cls(cls, vllm_config):
                model_path = vllm_config.model_config.kv_cache_config_builder_cls
                return model_path or DEFAULT_PATH

        cfg = _make_vllm_config(builder_cls_path=CUSTOM_PATH)
        assert ModelFirstPlatform.get_kv_cache_config_builder_cls(cfg) == CUSTOM_PATH


class TestDefaultBuilderDelegation:
    """Without a custom builder, the methods hit the default builder, which
    implements the planning steps in :mod:`kv_cache_planning`."""

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_kv_cache_configs")
    def test_get_kv_cache_configs_delegates_to_default(self, mock_impl, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        specs, memory = [MagicMock()], [0]
        builder = get_kv_cache_config_builder(cfg)
        result = builder.get_kv_cache_configs(cfg, specs, memory)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, specs, memory)
