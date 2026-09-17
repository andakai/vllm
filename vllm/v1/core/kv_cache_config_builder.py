# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable KV cache configuration builders."""

from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.kv_cache_plan import (
    KVCachePlanProvider,
    resolve_kv_cache_plan_provider,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


class KVCacheConfigBuilder:
    """Build final or fixed-capacity profiling cache configurations."""

    def __init__(self, plan_provider: KVCachePlanProvider | None = None) -> None:
        self.plan_provider = plan_provider

    def build(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
        *,
        fixed_num_blocks: int | None = None,
    ) -> list["KVCacheConfig"]:
        from vllm.v1.core.kv_cache_utils import _get_kv_cache_configs

        return _get_kv_cache_configs(
            vllm_config,
            kv_cache_specs,
            available_memory,
            plan_provider=self.plan_provider,
            fixed_num_blocks=fixed_num_blocks,
        )


def resolve_kv_cache_config_builder(
    vllm_config: "VllmConfig",
) -> KVCacheConfigBuilder:
    """Resolve a fresh model-aware builder and let the platform compose it."""
    from vllm.platforms import current_platform

    builder_path = vllm_config.model_config.kv_cache_config_builder_cls
    if builder_path is None:
        delegate = KVCacheConfigBuilder(resolve_kv_cache_plan_provider(vllm_config))
    else:
        builder_cls = resolve_obj_by_qualname(builder_path)
        delegate = builder_cls()
    return current_platform.get_kv_cache_config_builder(vllm_config, delegate)


def build_kv_cache_configs(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, "KVCacheSpec"]],
    available_memory: list[int],
    *,
    fixed_num_blocks: int | None = None,
) -> list["KVCacheConfig"]:
    return resolve_kv_cache_config_builder(vllm_config).build(
        vllm_config,
        kv_cache_specs,
        available_memory,
        fixed_num_blocks=fixed_num_blocks,
    )


def build_profiling_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, "KVCacheSpec"],
    min_blocks: int,
) -> "KVCacheConfig":
    return build_kv_cache_configs(
        vllm_config,
        [kv_cache_spec],
        [0],
        fixed_num_blocks=min_blocks,
    )[0]
