# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache config builder.

Resolution order:
    platform override > model declaration > default

The default builder simply forwards to :mod:`vllm.v1.core.kv_cache_planning`.
Platforms or models can subclass :class:`KVCacheConfigBuilder` to customize
KV cache planning end-to-end.

All KV cache planning entry points live here so that callers (engine core,
workers, HiSparse layout) never import the planner directly: a builder that
exists precisely because the default planner cannot handle its spec
combination must be consulted on every planning path, including the
profiling-time minimal config.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheLayout,
        KVCacheSpec,
    )


class KVCacheConfigBuilder:
    """Strategy class for model- or platform-specific KV cache planning."""

    def build_kv_cache_configs(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        """Return the per-worker KV cache configs.

        The default implementation delegates to the standard planning logic
        in :mod:`vllm.v1.core.kv_cache_planning`. Subclasses may override
        this to provide platform- or model-specific layouts.
        """
        from vllm.v1.core.kv_cache_planning import get_kv_cache_configs

        return get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)

    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"]:
        """Split a worker's layers into KV cache groups.

        Mirrors
        :func:`vllm.v1.core.kv_cache_planning.get_kv_cache_groups`.
        """
        from vllm.v1.core.kv_cache_planning import get_kv_cache_groups as _impl

        return _impl(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        available_memory: int,
    ) -> "KVCacheConfig":
        """Build a KV cache config from groups and available memory.

        Mirrors
        :func:`vllm.v1.core.kv_cache_planning.get_kv_cache_config_from_groups`.
        """
        from vllm.v1.core.kv_cache_planning import (
            get_kv_cache_config_from_groups as _impl,
        )

        return _impl(vllm_config, kv_cache_groups, available_memory)

    def validate_kv_cache_layout(
        self,
        layout: "KVCacheLayout",
        kv_cache_groups: list["KVCacheGroupSpec"],
    ) -> None:
        """Validate that a resolved layout can express the groups' packing.

        Mirrors
        :func:`vllm.v1.core.kv_cache_planning.validate_kv_cache_layout`.
        """
        from vllm.v1.core.kv_cache_planning import validate_kv_cache_layout as _impl

        _impl(layout, kv_cache_groups)

    def may_override_num_blocks(
        self, vllm_config: "VllmConfig", num_blocks: int
    ) -> int:
        """Apply `num_gpu_blocks_override` if set.

        Mirrors
        :func:`vllm.v1.core.kv_cache_planning.may_override_num_blocks`.
        """
        from vllm.v1.core.kv_cache_planning import may_override_num_blocks as _impl

        return _impl(vllm_config, num_blocks)

    def _get_kv_cache_bytes_per_block(
        self, kv_cache_groups: list["KVCacheGroupSpec"]
    ) -> int:
        """Return the largest cache group's bytes per block.

        Mirrors
        :func:`vllm.v1.core.kv_cache_planning._get_kv_cache_bytes_per_block`.
        """
        from vllm.v1.core.kv_cache_planning import (
            _get_kv_cache_bytes_per_block as _impl,
        )

        return _impl(kv_cache_groups)


def _load_builder(cls_path: str) -> KVCacheConfigBuilder:
    """Import and instantiate a builder from its fully-qualified class path."""
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls()


_BUILDER: KVCacheConfigBuilder | None = None
_BUILDER_KEY: str | None = None


def resolve_builder(vllm_config: "VllmConfig") -> KVCacheConfigBuilder:
    """Return the active KV cache config builder (process-wide singleton).

    Priority: platform override > model declaration > default. The resolved
    builder is cached on first use so every planning path in the process
    (engine core, workers, profiling) consults the same builder.
    """
    global _BUILDER, _BUILDER_KEY

    from vllm.platforms import current_platform

    platform_cls_path = current_platform.get_kv_cache_config_builder_cls(vllm_config)
    builder_key = (
        platform_cls_path or vllm_config.model_config.kv_cache_config_builder_cls
    )
    if _BUILDER is not None and builder_key == _BUILDER_KEY:
        return _BUILDER

    if platform_cls_path is not None:
        builder = _load_builder(platform_cls_path)
    elif builder_key is not None:
        builder = _load_builder(builder_key)
    else:
        builder = KVCacheConfigBuilder()

    _BUILDER = builder
    _BUILDER_KEY = builder_key
    return builder


def build_kv_cache_configs(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, "KVCacheSpec"]],
    available_memory: list[int],
) -> list["KVCacheConfig"]:
    """Resolve the active builder and delegate KV cache planning to it."""
    return resolve_builder(vllm_config).build_kv_cache_configs(
        vllm_config, kv_cache_specs, available_memory
    )


def get_kv_cache_groups(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, "KVCacheSpec"],
) -> list["KVCacheGroupSpec"]:
    """Resolve the active builder and split the layers into groups."""
    return resolve_builder(vllm_config).get_kv_cache_groups(vllm_config, kv_cache_spec)


def get_kv_cache_config_from_groups(
    vllm_config: "VllmConfig",
    kv_cache_groups: list["KVCacheGroupSpec"],
    available_memory: int,
) -> "KVCacheConfig":
    """Resolve the active builder and build a config from the given groups."""
    return resolve_builder(vllm_config).get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    )


def validate_kv_cache_layout(
    layout: "KVCacheLayout",
    kv_cache_groups: list["KVCacheGroupSpec"],
    vllm_config: "VllmConfig | None" = None,
) -> None:
    """Resolve the active builder and validate the layout for the groups."""
    config = vllm_config if vllm_config is not None else _builder_vllm_config()
    return resolve_builder(config).validate_kv_cache_layout(layout, kv_cache_groups)


def may_override_num_blocks(vllm_config: "VllmConfig", num_blocks: int) -> int:
    """Resolve the active builder and apply the num_gpu_blocks_override."""
    return resolve_builder(vllm_config).may_override_num_blocks(vllm_config, num_blocks)


def _get_kv_cache_bytes_per_block(
    kv_cache_groups: list["KVCacheGroupSpec"],
    vllm_config: "VllmConfig | None" = None,
) -> int:
    """Resolve the active builder and return the largest group's bytes per block."""
    config = vllm_config if vllm_config is not None else _builder_vllm_config()
    return resolve_builder(config)._get_kv_cache_bytes_per_block(kv_cache_groups)


def _builder_vllm_config() -> "VllmConfig":
    """Return the VllmConfig of the current vLLM context.

    The context is set by the engine/worker entry points, so the handful of
    planning helpers that take no vllm_config argument can still resolve the
    active builder.
    """
    from vllm.config import get_current_vllm_config

    return get_current_vllm_config()
