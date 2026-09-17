# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Declarative KV cache planning extensions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


@dataclass(frozen=True)
class KVCachePoolRegion:
    """One physical region per block, shared by all listed layers."""

    size_bytes: int
    layers: tuple[str, ...]


KVCachePoolPlan: TypeAlias = tuple[KVCachePoolRegion, ...]


class KVCachePlanProvider:
    """Describes ordinary model grouping and packing for the core builder."""

    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: Mapping[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"] | None:
        return None

    def get_kv_cache_regions(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: tuple["KVCacheGroupSpec", ...],
    ) -> KVCachePoolPlan | None:
        return None


def resolve_kv_cache_plan_provider(
    vllm_config: "VllmConfig",
) -> KVCachePlanProvider | None:
    """Resolve a fresh model plan provider for one builder."""
    provider_path = vllm_config.model_config.kv_cache_plan_provider_cls
    if provider_path is None:
        return None
    provider_cls = resolve_obj_by_qualname(provider_path)
    return provider_cls()
