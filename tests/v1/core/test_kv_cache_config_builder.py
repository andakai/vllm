# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import Platform
from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    _get_profiling_kv_cache_config,
)
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig


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
    KVCacheConfigBuilder._active = None
    ExactBlocksBuilder.seen_num_blocks = None
    yield
    KVCacheConfigBuilder._active = None
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
        assert type(KVCacheConfigBuilder._resolve(cfg)) is DefaultKVCacheConfigBuilder

    @patch("vllm.platforms.current_platform")
    def test_model_declared_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert isinstance(KVCacheConfigBuilder._resolve(cfg), CustomBuilder)

    @patch("vllm.platforms.current_platform")
    def test_resolves_once_and_caches(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        assert KVCacheConfigBuilder._resolve(cfg) is KVCacheConfigBuilder._resolve(cfg)

    @patch("vllm.platforms.current_platform")
    def test_reset_forces_resolution_again(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        first = KVCacheConfigBuilder._resolve(cfg)
        KVCacheConfigBuilder._active = None
        second = KVCacheConfigBuilder._resolve(cfg)
        assert first is not second
        assert isinstance(second, CustomBuilder)

    @patch("vllm.platforms.current_platform")
    def test_entry_points_delegate_to_resolved_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = CUSTOM_PATH
        cfg = _make_vllm_config()
        KVCacheConfigBuilder._resolve(cfg)
        active = KVCacheConfigBuilder._active
        assert isinstance(active, CustomBuilder)
        with patch.object(active, "get_kv_cache_configs", return_value=[]) as g:
            assert KVCacheConfigBuilder.get_kv_cache_configs(cfg, [], [0]) == []
            g.assert_called_once()

    def test_facade_only_exposes_core_entry_point(self):
        assert not hasattr(KVCacheConfigBuilder, "get_profiling_kv_cache_config")
        assert not hasattr(KVCacheConfigBuilder, "get_kv_cache_groups")
        assert not hasattr(KVCacheConfigBuilder, "get_pool_bytes_per_block")
        assert not hasattr(KVCacheConfigBuilder, "get_kv_cache_config_from_groups")

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
            assert isinstance(KVCacheConfigBuilder._resolve(cfg), CustomBuilder)

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
        result = KVCacheConfigBuilder.get_kv_cache_configs(cfg, specs, memory)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, specs, memory)
