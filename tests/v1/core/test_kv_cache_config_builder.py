# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.core.kv_cache_config_builder import (
    KVCacheConfigBuilder,
    _load_builder,
    resolve_builder,
)


def _make_vllm_config(builder_cls_path: str | None = None) -> MagicMock:
    """Create a minimal mock VllmConfig for builder resolution tests."""
    cfg = MagicMock()
    cfg.model_config.kv_cache_config_builder_cls = builder_cls_path
    return cfg


class CustomBuilder(KVCacheConfigBuilder):
    """A test builder subclass."""

    pass


class TestLoadBuilder:
    def test_load_default_builder(self):
        builder = _load_builder(
            "vllm.v1.core.kv_cache_config_builder.KVCacheConfigBuilder"
        )
        assert isinstance(builder, KVCacheConfigBuilder)

    def test_load_nonexistent_raises(self):
        with pytest.raises((ImportError, AttributeError)):
            _load_builder("vllm.nonexistent.module.Builder")


class TestResolveBuilder:
    @patch("vllm.platforms.current_platform")
    def test_default_when_no_overrides(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        cfg = _make_vllm_config(builder_cls_path=None)
        builder = resolve_builder(cfg)
        assert type(builder) is KVCacheConfigBuilder

    @patch("vllm.platforms.current_platform")
    def test_model_declared_builder(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        cls_path = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
        cfg = _make_vllm_config(builder_cls_path=cls_path)
        builder = resolve_builder(cfg)
        assert isinstance(builder, CustomBuilder)

    @patch("vllm.platforms.current_platform")
    def test_glm5_model_declared_builder(self, mock_platform):
        from vllm.models.glm5next.kv_cache_config import (
            Glm5NextKVCacheConfigBuilder,
        )

        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        cfg = _make_vllm_config(
            "vllm.models.glm5next.kv_cache_config."
            "Glm5NextKVCacheConfigBuilder"
        )
        assert isinstance(resolve_builder(cfg), Glm5NextKVCacheConfigBuilder)

    @patch("vllm.platforms.current_platform")
    def test_platform_overrides_model(self, mock_platform):
        platform_path = "tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
        mock_platform.get_kv_cache_config_builder_cls.return_value = platform_path
        # Model declares default, but platform overrides
        cfg = _make_vllm_config(builder_cls_path=None)
        builder = resolve_builder(cfg)
        assert isinstance(builder, CustomBuilder)


class TestBuilderSingleton:
    def setup_method(self):
        import vllm.v1.core.kv_cache_config_builder as builder_mod

        builder_mod._BUILDER = None
        builder_mod._BUILDER_KEY = None

    def teardown_method(self):
        import vllm.v1.core.kv_cache_config_builder as builder_mod

        builder_mod._BUILDER = None
        builder_mod._BUILDER_KEY = None

    @patch("vllm.platforms.current_platform")
    def test_returns_cached_builder_for_same_key(self, mock_platform):
        import vllm.v1.core.kv_cache_config_builder as builder_mod

        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        cfg = _make_vllm_config(builder_cls_path=None)
        first = resolve_builder(cfg)
        second = resolve_builder(cfg)
        assert first is second
        assert builder_mod._BUILDER is first

    @patch("vllm.platforms.current_platform")
    def test_reloads_when_model_builder_changes(self, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        default_cfg = _make_vllm_config(builder_cls_path=None)
        custom_cfg = _make_vllm_config(
            builder_cls_path="tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
        )
        assert type(resolve_builder(default_cfg)) is KVCacheConfigBuilder
        assert type(resolve_builder(custom_cfg)) is CustomBuilder
        assert type(resolve_builder(default_cfg)) is KVCacheConfigBuilder

    @patch("vllm.platforms.current_platform")
    def test_planning_entry_points_use_the_builder(self, mock_platform):
        import vllm.v1.core.kv_cache_config_builder as builder_mod

        builder = MagicMock()
        builder.build_kv_cache_configs.return_value = []
        builder.get_kv_cache_groups.return_value = []
        mock_platform.get_kv_cache_config_builder_cls.return_value = None
        cfg = _make_vllm_config(
            builder_cls_path="tests.v1.core.test_kv_cache_config_builder.CustomBuilder"
        )
        with (
            patch.object(builder_mod, "_load_builder", return_value=builder) as load,
            patch.object(builder_mod, "_BUILDER", None),
            patch.object(builder_mod, "_BUILDER_KEY", None),
        ):
            assert builder_mod.build_kv_cache_configs(cfg, [], [0]) == []
            builder.build_kv_cache_configs.assert_called_once()
            assert builder_mod.get_kv_cache_groups(cfg, {}) == []
            builder.get_kv_cache_groups.assert_called_once()
            # Second call reuses the cached builder without re-loading.
            assert builder_mod.build_kv_cache_configs(cfg, [], [0]) == []
            assert load.call_count == 1


def test_glm5_models_declare_kv_cache_builder():
    from vllm.model_executor.models.registry import _ModelInfo
    from vllm.models.glm5next import (
        Glm5NextForCausalLM,
        Glm5NextForConditionalGeneration,
    )

    expected = (
        "vllm.models.glm5next.kv_cache_config.Glm5NextKVCacheConfigBuilder"
    )
    for model_cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert model_cls.kv_cache_config_builder_cls == expected
        model_info = _ModelInfo.from_model_cls(model_cls)
        assert model_info.kv_cache_config_builder_cls == expected


def test_glm5_builder_preserves_generic_grouping_precedence():
    from vllm.models.glm5next.kv_cache_config import (
        Glm5NextKVCacheConfigBuilder,
    )

    cfg = _make_vllm_config()
    cfg.scheduler_config.disable_hybrid_kv_cache_manager = True
    cfg.attention_config.hisparse_config = None
    builder = Glm5NextKVCacheConfigBuilder()

    with patch.object(
        KVCacheConfigBuilder, "get_kv_cache_groups", return_value=[]
    ) as default_grouping:
        assert builder.get_kv_cache_groups(cfg, {}) == []

    default_grouping.assert_called_once_with(cfg, {})
