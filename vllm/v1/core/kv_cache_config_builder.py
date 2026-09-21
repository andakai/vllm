# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache config builder resolution."""

from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


class KVCacheConfigBuilder:
    """Resolve and invoke the active KV cache config builder.

    Resolution priority is owned by the platform hook
    (:meth:`vllm.platforms.interface.Platform.get_kv_cache_config_builder_cls`);
    this class only caches the resolved builder and exposes the Core planning
    entry point. Model and platform builders customize the three hooks on
    :class:`DefaultKVCacheConfigBuilder` instead.
    """

    _active: "DefaultKVCacheConfigBuilder | None" = None

    @classmethod
    def _resolve(cls, vllm_config: "VllmConfig") -> "DefaultKVCacheConfigBuilder":
        if cls._active is None:
            from vllm.platforms import current_platform

            builder_cls = resolve_obj_by_qualname(
                current_platform.get_kv_cache_config_builder_cls(vllm_config)
            )
            cls._active = builder_cls()
        return cls._active

    @classmethod
    def get_kv_cache_configs(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        """Generate the full KV cache configurations for every worker.

        The main entry point: takes the per-worker KV cache specs and the
        memory available on each worker, runs the whole planning pipeline
        (merge specs, group layers, project to workers, auto-fit
        max_model_len, admission checks, per-worker layouts, min-blocks
        convergence), and returns one ready-to-allocate
        :class:`~vllm.v1.kv_cache_interface.KVCacheConfig` per worker.

        Consumed by ``vllm/v1/engine/core.py``.
        """
        return cls._resolve(vllm_config).get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory
        )


def _get_profiling_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, "KVCacheSpec"],
    min_blocks: int,
) -> "KVCacheConfig":
    """Build profiling storage through the active builder's normal hooks."""
    builder = KVCacheConfigBuilder._resolve(vllm_config)
    groups = builder.get_kv_cache_groups(vllm_config, kv_cache_spec)
    return builder.get_kv_cache_config_from_groups(
        vllm_config, groups, max(min_blocks, 1)
    )
