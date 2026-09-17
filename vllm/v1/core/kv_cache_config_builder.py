# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache configuration planning."""

from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )


class KVCacheConfigBuilder:
    """Default KV cache planner and extension point for models and platforms."""

    def build_kv_cache_configs(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        return get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_memory, builder=self
        )

    def build_profiling_kv_cache_config(
        self,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
        min_blocks: int,
    ) -> "KVCacheConfig":
        groups = self.get_kv_cache_groups(vllm_config, kv_cache_spec)
        cache_config = vllm_config.cache_config
        saved_override = cache_config.num_gpu_blocks_override
        cache_config.num_gpu_blocks_override = min_blocks
        try:
            return self.get_kv_cache_config_from_groups(
                vllm_config, groups, available_memory=0
            )
        finally:
            cache_config.num_gpu_blocks_override = saved_override

    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"]:
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        return get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        available_memory: int,
    ) -> "KVCacheConfig":
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

        return get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory, builder=self
        )

    def get_kv_cache_bytes_per_block(
        self, kv_cache_groups: list["KVCacheGroupSpec"]
    ) -> int:
        from vllm.v1.core.kv_cache_utils import _get_kv_cache_bytes_per_block

        return _get_kv_cache_bytes_per_block(kv_cache_groups)

    def max_memory_usage_bytes_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
    ) -> int:
        from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups

        return _max_memory_usage_bytes_from_groups(
            vllm_config, kv_cache_groups, builder=self
        )


def resolve_builder(vllm_config: "VllmConfig") -> KVCacheConfigBuilder:
    """Resolve a fresh builder with platform > model > default precedence."""
    from vllm.platforms import current_platform

    builder_cls_path = current_platform.get_kv_cache_config_builder_cls(vllm_config)
    if builder_cls_path is None:
        builder_cls_path = vllm_config.model_config.kv_cache_config_builder_cls
    if builder_cls_path is None:
        return KVCacheConfigBuilder()
    builder_cls = resolve_obj_by_qualname(builder_cls_path)
    return builder_cls()


def build_kv_cache_configs(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, "KVCacheSpec"]],
    available_memory: list[int],
) -> list["KVCacheConfig"]:
    builder = resolve_builder(vllm_config)
    return builder.build_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)


def build_profiling_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, "KVCacheSpec"],
    min_blocks: int,
) -> "KVCacheConfig":
    builder = resolve_builder(vllm_config)
    return builder.build_profiling_kv_cache_config(
        vllm_config, kv_cache_spec, min_blocks
    )
