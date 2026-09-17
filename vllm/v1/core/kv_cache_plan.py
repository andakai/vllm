# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planning extension points."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class KVCacheGroupPlanEntry:
    """One cache group expressed as immutable per-layer specs."""

    layer_specs: tuple[tuple[str, KVCacheSpec], ...]
    use_uniform_type: bool = False


@dataclass(frozen=True)
class KVCacheGroupPlan:
    groups: tuple[KVCacheGroupPlanEntry, ...]


@dataclass(frozen=True)
class KVCachePoolRegion:
    """One physical region per block, shared by all listed layers."""

    size_bytes: int
    layers: tuple[str, ...]


class KVCacheWorkerPlan:
    """Executable per-worker cache accounting and placement."""

    @property
    def bytes_per_block(self) -> int:
        raise NotImplementedError

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        raise NotImplementedError

    def materialize(self, vllm_config: "VllmConfig", num_blocks: int) -> KVCacheConfig:
        raise NotImplementedError


def _get_layer_specs(
    kv_cache_groups: tuple[KVCacheGroupSpec, ...],
) -> dict[str, KVCacheSpec]:
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            layer_specs.update(
                (name, group_spec.kv_cache_specs[name]) for name in group.layer_names
            )
        else:
            layer_specs.update(dict.fromkeys(group.layer_names, group_spec))
    return layer_specs


class RegionWorkerPlan(KVCacheWorkerPlan):
    """Executable adapter for block-compact declarative regions."""

    def __init__(
        self,
        kv_cache_groups: tuple[KVCacheGroupSpec, ...],
        regions: tuple[KVCachePoolRegion, ...],
    ) -> None:
        self.kv_cache_groups = kv_cache_groups
        self.regions = regions
        layer_groups = {
            layer_name: group_index
            for group_index, group in enumerate(kv_cache_groups)
            for layer_name in group.layer_names
        }
        layer_specs = _get_layer_specs(kv_cache_groups)
        planned_layers = [layer for region in regions for layer in region.layers]
        if len(planned_layers) != len(set(planned_layers)):
            raise ValueError("KV cache regions assign a layer more than once")
        if set(planned_layers) != set(layer_specs):
            raise ValueError("KV cache regions must assign every projected layer")
        for region in regions:
            if not region.layers or region.size_bytes <= 0:
                raise ValueError("KV cache regions must be non-empty and positive")
            if any(
                layer_specs[layer_name].page_size_bytes != region.size_bytes
                for layer_name in region.layers
            ):
                raise ValueError("KV cache layer page must match its region")
            region_groups = {layer_groups[layer_name] for layer_name in region.layers}
            if len(region_groups) != len(region.layers):
                raise ValueError(
                    "KV cache layers in one group cannot alias the same region"
                )
        self._layer_specs = layer_specs
        self._bytes_per_block = sum(region.size_bytes for region in regions)

    @property
    def bytes_per_block(self) -> int:
        return self._bytes_per_block

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        total_blocks = 0
        for group in self.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                total_blocks += spec.max_memory_usage_pages(vllm_config)
            else:
                total_blocks += cdiv(
                    spec.max_memory_usage_bytes(vllm_config),
                    spec.page_size_bytes,
                )
        return self.bytes_per_block * total_blocks

    def materialize(self, vllm_config: "VllmConfig", num_blocks: int) -> KVCacheConfig:
        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not layout.is_block_compact:
            raise ValueError(
                "Declarative KV cache regions require a block-compact "
                f"layout, but got {layout.name}."
            )
        size = self.bytes_per_block * num_blocks
        tensors: list[KVCacheTensor] = []
        region_offset = 0
        for region in self.regions:
            offset = region_offset * num_blocks
            for layer_name in region.layers:
                spec = self._layer_specs[layer_name]
                tensors.append(
                    KVCacheTensor(
                        size=size,
                        layers=[layer_name],
                        layer_stride=spec.page_size_bytes * num_blocks,
                        block_stride=spec.page_size_bytes,
                        offset=offset,
                    )
                )
            region_offset += region.size_bytes
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=list(self.kv_cache_groups),
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )


class KVCachePlanProvider:
    """Provides global grouping and opaque per-worker executable plans."""

    def __init__(self, delegate: "KVCachePlanProvider | None" = None) -> None:
        self._delegate = delegate

    def get_group_plan(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: Mapping[str, KVCacheSpec],
    ) -> KVCacheGroupPlan | None:
        if self._delegate is not None:
            return self._delegate.get_group_plan(vllm_config, kv_cache_specs)
        return None

    def get_worker_plan(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: tuple[KVCacheGroupSpec, ...],
    ) -> KVCacheWorkerPlan | None:
        if self._delegate is not None:
            return self._delegate.get_worker_plan(vllm_config, kv_cache_groups)
        return None


def resolve_kv_cache_plan_provider(vllm_config: "VllmConfig") -> KVCachePlanProvider:
    """Build a fresh platform > model > default provider chain."""
    from vllm.platforms import current_platform

    provider = KVCachePlanProvider()
    model_path = vllm_config.model_config.kv_cache_plan_provider_cls
    if model_path is not None:
        provider_cls = resolve_obj_by_qualname(model_path)
        provider = provider_cls(provider)
    platform_path = current_platform.get_kv_cache_plan_provider_cls(vllm_config)
    if platform_path is not None:
        provider_cls = resolve_obj_by_qualname(platform_path)
        provider = provider_cls(provider)
    return provider


def get_profiling_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, KVCacheSpec],
    min_blocks: int,
) -> KVCacheConfig:
    """Build a profiling cache through the same planning pipeline as startup."""
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_configs

    return _get_kv_cache_configs(
        vllm_config,
        [kv_cache_spec],
        [0],
        fixed_num_blocks=min_blocks,
        check_capacity=False,
    )[0]
