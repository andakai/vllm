# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative KV cache planning extensions."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )


@dataclass(frozen=True)
class KVCacheGroupPlanEntry:
    """One cache group expressed as immutable per-layer specs."""

    layer_specs: tuple[tuple[str, "KVCacheSpec"], ...]
    use_uniform_type: bool = False


@dataclass(frozen=True)
class KVCacheGroupPlan:
    groups: tuple[KVCacheGroupPlanEntry, ...]


@dataclass(frozen=True)
class KVCachePoolRegion:
    """One physical region per block, shared by all listed layers."""

    size_bytes: int
    layers: tuple[str, ...]


@dataclass(frozen=True)
class KVCachePoolPlan:
    regions: tuple[KVCachePoolRegion, ...]


class KVCachePlanProvider:
    """Produces immutable plans; core owns validation and materialization."""

    def get_group_plan(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: Mapping[str, "KVCacheSpec"],
    ) -> KVCacheGroupPlan | None:
        return None

    def get_pool_plan(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: tuple["KVCacheGroupSpec", ...],
    ) -> KVCachePoolPlan | None:
        return None


def resolve_kv_cache_plan_provider(
    vllm_config: "VllmConfig",
) -> KVCachePlanProvider | None:
    """Resolve a fresh provider with platform > model > default precedence."""
    from vllm.platforms import current_platform

    provider_path = current_platform.get_kv_cache_plan_provider_cls(vllm_config)
    if provider_path is None:
        provider_path = vllm_config.model_config.kv_cache_plan_provider_cls
    if provider_path is None:
        return None
    provider_cls = resolve_obj_by_qualname(provider_path)
    return provider_cls()


def get_profiling_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, "KVCacheSpec"],
    min_blocks: int,
) -> "KVCacheConfig":
    """Build the minimal profiling cache through the selected plan provider."""
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
        materialize_kv_cache_group_plan,
    )

    provider = resolve_kv_cache_plan_provider(vllm_config)
    group_plan = (
        provider.get_group_plan(vllm_config, MappingProxyType(kv_cache_spec))
        if provider is not None
        else None
    )
    groups = (
        materialize_kv_cache_group_plan(kv_cache_spec, group_plan)
        if group_plan is not None
        else get_kv_cache_groups(vllm_config, kv_cache_spec)
    )
    pool_plan = (
        provider.get_pool_plan(vllm_config, tuple(groups))
        if provider is not None
        else None
    )
    cache_config = vllm_config.cache_config
    saved_override = cache_config.num_gpu_blocks_override
    cache_config.num_gpu_blocks_override = min_blocks
    try:
        return get_kv_cache_config_from_groups(
            vllm_config, groups, available_memory=0, pool_plan=pool_plan
        )
    finally:
        cache_config.num_gpu_blocks_override = saved_override
