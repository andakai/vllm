# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheConfigBuilder resolution."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import Platform, current_platform
from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder


def _skip_glm5_on_xpu():
    if current_platform.is_xpu():
        pytest.skip("GLM-5.3-Flash does not support XPU")


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


@pytest.fixture(autouse=True)
def _reset_active_builder():
    KVCacheConfigBuilder.reset()
    yield
    KVCacheConfigBuilder.reset()


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

    def test_glm5_model_declared_builder(self):
        _skip_glm5_on_xpu()
        from vllm.models.glm5next.kv_cache_config import (
            Glm5NextKVCacheConfigBuilder,
        )

        cfg = _make_vllm_config(
            "vllm.models.glm5next.kv_cache_config."
            "Glm5NextKVCacheConfigBuilder"
        )
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_kv_cache_config_builder_cls.return_value = (
                cfg.model_config.kv_cache_config_builder_cls
            )
            assert isinstance(
                KVCacheConfigBuilder._resolve(cfg), Glm5NextKVCacheConfigBuilder
            )

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
        KVCacheConfigBuilder.reset()
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
        with (
            patch.object(active, "get_kv_cache_configs", return_value=[]) as g,
            patch.object(active, "get_kv_cache_groups", return_value=[]) as h,
        ):
            assert KVCacheConfigBuilder.get_kv_cache_configs(cfg, [], [0]) == []
            g.assert_called_once()
            assert KVCacheConfigBuilder.get_kv_cache_groups(cfg, {}) == []
            h.assert_called_once()


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
    @patch.object(DefaultKVCacheConfigBuilder, "get_kv_cache_groups")
    def test_get_kv_cache_groups_delegates_to_default(self, mock_impl, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        spec = {"layer": MagicMock()}
        assert (
            KVCacheConfigBuilder.get_kv_cache_groups(cfg, spec)
            is mock_impl.return_value
        )
        mock_impl.assert_called_once_with(cfg, spec)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_kv_cache_config_from_groups")
    def test_get_kv_cache_config_from_groups_delegates_to_default(
        self, mock_impl, mock_platform
    ):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        groups = [MagicMock()]
        result = KVCacheConfigBuilder.get_kv_cache_config_from_groups(cfg, groups, 0)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, groups, 0)

    @patch("vllm.platforms.current_platform")
    @patch.object(DefaultKVCacheConfigBuilder, "get_kv_cache_configs")
    def test_get_kv_cache_configs_delegates_to_default(self, mock_impl, mock_platform):
        mock_platform.get_kv_cache_config_builder_cls.return_value = DEFAULT_PATH
        cfg = _make_vllm_config()
        specs, memory = [MagicMock()], [0]
        result = KVCacheConfigBuilder.get_kv_cache_configs(cfg, specs, memory)
        assert result is mock_impl.return_value
        mock_impl.assert_called_once_with(cfg, specs, memory)


def test_glm5_models_declare_kv_cache_builder():
    _skip_glm5_on_xpu()
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
    _skip_glm5_on_xpu()
    from vllm.models.glm5next.kv_cache_config import (
        Glm5NextKVCacheConfigBuilder,
    )

    cfg = _make_vllm_config()
    cfg.scheduler_config.disable_hybrid_kv_cache_manager = True
    cfg.attention_config.hisparse_config = None
    builder = Glm5NextKVCacheConfigBuilder()

    with patch.object(
        DefaultKVCacheConfigBuilder, "get_kv_cache_groups", return_value=[]
    ) as default_grouping:
        assert builder.get_kv_cache_groups(cfg, {}) == []

    default_grouping.assert_called_once_with(cfg, {})
