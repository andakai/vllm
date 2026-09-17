# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_plan import (
    KVCacheGroupPlan,
    KVCacheGroupPlanEntry,
    KVCachePlanProvider,
    KVCachePoolPlan,
    KVCachePoolRegion,
)
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)


def _pp_balanced_mamba_group_count(
    vllm_config: VllmConfig,
    mamba_layer_names: list[str],
    mla_layer_names: list[str],
) -> int | None:
    num_groups = cdiv(len(mamba_layer_names), len(mla_layer_names))
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    if pp_size == 1:
        return num_groups

    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.models.utils import extract_layer_index

    total_layers = vllm_config.model_config.get_total_num_hidden_layers()
    mamba_indices = [extract_layer_index(name) for name in mamba_layer_names]
    mla_indices = [extract_layer_index(name) for name in mla_layer_names]
    for rank in range(pp_size):
        start, end = get_pp_indices(total_layers, rank, pp_size)
        num_mamba = sum(start <= index < end for index in mamba_indices)
        num_mla = sum(start <= index < end for index in mla_indices)
        if not num_mamba:
            continue
        if not num_mla:
            return None
        num_groups = max(num_groups, cdiv(num_mamba, num_mla))
    return num_groups


def _get_glm5_group_plan(
    vllm_config: VllmConfig,
    kv_cache_specs: Mapping[str, KVCacheSpec],
) -> KVCacheGroupPlan | None:
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if isinstance(spec, MambaSpec)
    }
    tail_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if isinstance(spec, KpoolTailSpec)
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attn_specs.values()
    ):
        return None

    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
    idx_pages = {
        spec.page_size_bytes for spec in mla_specs.values() if spec.tokens_per_state > 1
    }
    if not idx_pages:
        return None
    assert all(spec.page_size_padded is None for spec in mla_specs.values())
    assert len(idx_pages) == 1
    mla_names = [name for name, spec in mla_specs.items() if spec.tokens_per_state == 1]
    mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
    assert len(mla_pages) == 1
    mla_page = mla_pages.pop()

    entries = [KVCacheGroupPlanEntry(tuple(attn_specs.items()), True)]
    if tail_specs:
        idx_page = next(iter(idx_pages))
        padded_tail_specs = tuple(
            (name, replace(spec, page_size_padded=idx_page))
            for name, spec in tail_specs.items()
        )
        entries.append(KVCacheGroupPlanEntry(padded_tail_specs, True))

    any_mamba = next(iter(mamba_specs.values()))
    assert all(spec == any_mamba for spec in mamba_specs.values())
    if any_mamba.real_page_size_bytes > mla_page:
        raise ValueError(
            f"the mamba state page ({any_mamba.real_page_size_bytes} bytes) "
            f"does not fit the MLA page ({mla_page} bytes); increase tensor "
            "parallelism or use a wider KV cache dtype"
        )
    padded_mamba = replace(any_mamba, page_size_padded=mla_page)
    num_groups = _pp_balanced_mamba_group_count(
        vllm_config, list(mamba_specs), mla_names
    )
    if num_groups is None:
        raise ValueError(
            "a pipeline stage has mamba layers but no MLA layer to share "
            "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
        )
    grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for index, name in enumerate(mamba_specs):
        grouped_names[index % num_groups].append(name)
    entries.extend(
        KVCacheGroupPlanEntry(tuple((name, padded_mamba) for name in layer_names))
        for layer_names in grouped_names
    )
    return KVCacheGroupPlan(tuple(entries))


def _glm5_layout(
    kv_cache_groups: tuple[KVCacheGroupSpec, ...],
) -> tuple[list[KVCacheGroupSpec], list[str], list[str], int, int, list[str]] | None:
    uniform_groups = [
        group
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    ]
    mamba_groups = [
        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):
            attn_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):
            tail_group = group
    if attn_group is None or not mamba_groups:
        return None
    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):
        return None

    attn_specs = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec).kv_cache_specs
    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in mla_specs.values()
    ):
        return None
    mla_names = [
        name for name in attn_group.layer_names if mla_specs[name].tokens_per_state == 1
    ]
    idx_names = [
        name for name in attn_group.layer_names if mla_specs[name].tokens_per_state > 1
    ]
    mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
    idx_pages = {mla_specs[name].page_size_bytes for name in idx_names}
    if len(mla_pages) != 1 or len(idx_pages) != 1:
        return None
    mla_page = mla_pages.pop()
    idx_page = idx_pages.pop()
    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None

    tail_names: list[str] = []
    if tail_group is not None:
        tail_names = list(tail_group.layer_names)
        tail_specs = cast(
            UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
        ).kv_cache_specs
        tail_pages = {
            cast(KpoolTailSpec, spec).unpadded_page_size_bytes
            for spec in tail_specs.values()
        }
        if (
            len(tail_pages) != 1
            or len(tail_names) != len(idx_names)
            or tail_pages.pop() > idx_page
        ):
            return None
    return mamba_groups, mla_names, idx_names, mla_page, idx_page, tail_names


class Glm5NextKVCachePlanProvider(KVCachePlanProvider):
    def get_group_plan(self, vllm_config, kv_cache_specs):
        if (
            vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            or vllm_config.attention_config.hisparse_config is not None
        ):
            return None
        return _get_glm5_group_plan(vllm_config, kv_cache_specs)

    def get_pool_plan(self, vllm_config, kv_cache_groups):
        if vllm_config.attention_config.hisparse_config is not None:
            return None
        layout = _glm5_layout(kv_cache_groups)
        if layout is None:
            return None
        mamba_groups, mla_names, idx_names, mla_page, idx_page, tail_names = layout
        regions = []
        for index, mla_name in enumerate(mla_names):
            aliases = [mla_name]
            aliases.extend(
                group.layer_names[index]
                for group in mamba_groups
                if index < len(group.layer_names)
            )
            regions.append(KVCachePoolRegion(mla_page, tuple(aliases)))
        for index, idx_name in enumerate(idx_names):
            indexer_aliases: tuple[str, ...] = (
                (idx_name, tail_names[index]) if tail_names else (idx_name,)
            )
            regions.append(KVCachePoolRegion(idx_page, indexer_aliases))
        return KVCachePoolPlan(tuple(regions))
