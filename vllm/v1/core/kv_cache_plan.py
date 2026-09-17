# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composable KV cache planning extensions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )


@dataclass(frozen=True)
class KVCachePoolRegion:
    """One physical region per block, shared by all listed layers."""

    size_bytes: int
    layers: tuple[str, ...]


KVCachePoolPlan: TypeAlias = tuple[KVCachePoolRegion, ...]
KVCacheGroupPlanner: TypeAlias = Callable[
    ["VllmConfig", Mapping[str, "KVCacheSpec"]],
    list["KVCacheGroupSpec"] | None,
]
KVCacheRegionPlanner: TypeAlias = Callable[
    ["VllmConfig", tuple["KVCacheGroupSpec", ...]],
    KVCachePoolPlan | None,
]


@dataclass(frozen=True)
class KVCachePlanningRequest:
    """One immutable request passed through the planning chain."""

    vllm_config: "VllmConfig"
    kv_cache_specs: list[dict[str, "KVCacheSpec"]]
    available_memory: list[int]
    fixed_num_blocks: int | None = None
    group_planner: KVCacheGroupPlanner | None = None
    region_planner: KVCacheRegionPlanner | None = None

    def with_declarative_plan(
        self,
        group_planner: KVCacheGroupPlanner,
        region_planner: KVCacheRegionPlanner,
    ) -> "KVCachePlanningRequest":
        return replace(
            self,
            group_planner=group_planner,
            region_planner=region_planner,
        )


KVCachePlanner: TypeAlias = Callable[[KVCachePlanningRequest], list["KVCacheConfig"]]


def _default_kv_cache_planner(
    request: KVCachePlanningRequest,
) -> list["KVCacheConfig"]:
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_configs

    return _get_kv_cache_configs(
        request.vllm_config,
        request.kv_cache_specs,
        request.available_memory,
        group_planner=request.group_planner,
        region_planner=request.region_planner,
        fixed_num_blocks=request.fixed_num_blocks,
    )


def resolve_kv_cache_planner(vllm_config: "VllmConfig") -> KVCachePlanner:
    """Compose a fresh platform > model > default planning chain."""
    from vllm.platforms import current_platform

    planner: KVCachePlanner = _default_kv_cache_planner
    planner_path = vllm_config.model_config.kv_cache_planner_cls
    if planner_path is not None:
        planner_cls = resolve_obj_by_qualname(planner_path)
        planner = planner_cls(planner)
    return current_platform.get_kv_cache_planner(vllm_config, planner)


def build_kv_cache_configs(
    vllm_config: "VllmConfig",
    kv_cache_specs: list[dict[str, "KVCacheSpec"]],
    available_memory: list[int],
    *,
    fixed_num_blocks: int | None = None,
) -> list["KVCacheConfig"]:
    request = KVCachePlanningRequest(
        vllm_config,
        kv_cache_specs,
        available_memory,
        fixed_num_blocks,
    )
    return resolve_kv_cache_planner(vllm_config)(request)


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
