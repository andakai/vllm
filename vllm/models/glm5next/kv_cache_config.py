# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from typing import cast

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

GlmLayout = tuple[
    KVCacheGroupSpec,
    list[KVCacheGroupSpec],
    list[str],
    list[str],
    int,
    int,
    list[str],
]


def _pp_balanced_mamba_group_count(
    vllm_config: VllmConfig,
    mamba_layer_names: list[str],
    mla_layer_names: list[str],
) -> int | None:
    """Choose enough Mamba groups for every PP stage to fit MLA slots."""
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


def _get_glm_groups(
    vllm_config: VllmConfig,
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build GLM MLA/indexer, tail, and Mamba scheduler groups."""
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
    attention_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if not isinstance(spec, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(spec) is MLAAttentionSpec for spec in attention_specs.values()
    ):
        return None

    mla_specs = cast(dict[str, MLAAttentionSpec], attention_specs)
    indexer_pages = {
        spec.page_size_bytes for spec in mla_specs.values() if spec.tokens_per_state > 1
    }
    if not indexer_pages:
        return None
    if any(spec.page_size_padded is not None for spec in mla_specs.values()):
        raise ValueError("GLM MLA specs must not already be page padded.")
    if len(indexer_pages) != 1:
        raise ValueError("GLM indexer specs must use one page size.")

    mla_names = [name for name, spec in mla_specs.items() if spec.tokens_per_state == 1]
    mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
    if len(mla_pages) != 1:
        raise ValueError("GLM MLA specs must use one page size.")
    mla_page = mla_pages.pop()

    attention_uniform = UniformTypeKVCacheSpecs.from_specs(attention_specs)
    if attention_uniform is None:
        raise ValueError("GLM MLA and indexer specs must share manager semantics.")
    groups = [KVCacheGroupSpec(list(attention_specs), attention_uniform)]

    if tail_specs:
        indexer_page = next(iter(indexer_pages))
        padded_tail_specs: dict[str, KVCacheSpec] = {
            name: replace(spec, page_size_padded=indexer_page)
            for name, spec in tail_specs.items()
        }
        tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
        if tail_uniform is None:
            raise ValueError("GLM tail specs must share manager semantics.")
        groups.append(KVCacheGroupSpec(list(padded_tail_specs), tail_uniform))

    any_mamba = next(iter(mamba_specs.values()))
    if not all(spec == any_mamba for spec in mamba_specs.values()):
        raise ValueError("GLM Mamba specs must be identical before padding.")
    if any_mamba.real_page_size_bytes > mla_page:
        raise ValueError(
            f"the mamba state page ({any_mamba.real_page_size_bytes} bytes) "
            f"does not fit the MLA page ({mla_page} bytes); increase tensor "
            "parallelism or use a wider KV cache dtype"
        )

    padded_mamba = replace(any_mamba, page_size_padded=mla_page)
    num_mamba_groups = _pp_balanced_mamba_group_count(
        vllm_config, list(mamba_specs), mla_names
    )
    if num_mamba_groups is None:
        raise ValueError(
            "a pipeline stage has mamba layers but no MLA layer to share "
            "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
        )
    grouped_mamba_names: list[list[str]] = [[] for _ in range(num_mamba_groups)]
    for index, name in enumerate(mamba_specs):
        grouped_mamba_names[index % num_mamba_groups].append(name)
    groups.extend(
        KVCacheGroupSpec(layer_names, padded_mamba)
        for layer_names in grouped_mamba_names
    )
    return groups


def _get_glm_layout(groups: list[KVCacheGroupSpec]) -> GlmLayout | None:
    """Recover GLM physical slots after optional PP projection."""
    uniform_groups = [
        group
        for group in groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    ]
    mamba_groups = [
        group for group in groups if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    attention_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    for group in uniform_groups:
        specs = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs
        if all(type(spec) is MLAAttentionSpec for spec in specs.values()):
            attention_group = group
        elif all(isinstance(spec, KpoolTailSpec) for spec in specs.values()):
            tail_group = group

    if attention_group is None or not mamba_groups:
        return None
    if len(uniform_groups) + len(mamba_groups) != len(groups):
        return None

    attention_specs = cast(
        UniformTypeKVCacheSpecs, attention_group.kv_cache_spec
    ).kv_cache_specs
    mla_specs = cast(dict[str, MLAAttentionSpec], attention_specs)
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in mla_specs.values()
    ):
        return None

    mla_names = [
        name
        for name in attention_group.layer_names
        if mla_specs[name].tokens_per_state == 1
    ]
    indexer_names = [
        name
        for name in attention_group.layer_names
        if mla_specs[name].tokens_per_state > 1
    ]
    mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
    indexer_pages = {mla_specs[name].page_size_bytes for name in indexer_names}
    if len(mla_pages) != 1 or len(indexer_pages) != 1:
        return None
    mla_page = mla_pages.pop()
    indexer_page = indexer_pages.pop()
    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None
    if any(len(group.layer_names) > len(mla_names) for group in mamba_groups):
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
        if len(tail_pages) != 1 or len(tail_names) != len(indexer_names):
            return None
        if tail_pages.pop() > indexer_page:
            return None

    return (
        attention_group,
        mamba_groups,
        mla_names,
        indexer_names,
        mla_page,
        indexer_page,
        tail_names,
    )


def _get_glm_pool_bytes_per_block(layout: GlmLayout) -> int:
    _, _, mla_names, indexer_names, mla_page, indexer_page, _ = layout
    return len(mla_names) * mla_page + len(indexer_names) * indexer_page


def _materialize_glm_config(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    layout: GlmLayout,
    num_blocks: int,
) -> KVCacheConfig:
    (
        attention_group,
        mamba_groups,
        mla_names,
        indexer_names,
        mla_page,
        indexer_page,
        tail_names,
    ) = layout
    backing_size = _get_glm_pool_bytes_per_block(layout) * num_blocks
    attention_specs = cast(
        UniformTypeKVCacheSpecs, attention_group.kv_cache_spec
    ).kv_cache_specs
    tensors: list[KVCacheTensor] = []

    def add_tensor(layer_name: str, spec: KVCacheSpec, offset: int) -> None:
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                layer_stride=spec.page_size_bytes * num_blocks,
                block_stride=spec.page_size_bytes,
                offset=offset,
            )
        )

    for slot, mla_name in enumerate(mla_names):
        offset = slot * mla_page * num_blocks
        add_tensor(mla_name, attention_specs[mla_name], offset)
        for group in mamba_groups:
            if slot < len(group.layer_names):
                add_tensor(group.layer_names[slot], group.kv_cache_spec, offset)

    indexer_base = len(mla_names) * mla_page * num_blocks
    tail_group = next(
        (
            group
            for group in groups
            if any(name in group.layer_names for name in tail_names)
        ),
        None,
    )
    tail_specs = (
        cast(UniformTypeKVCacheSpecs, tail_group.kv_cache_spec).kv_cache_specs
        if tail_group is not None
        else {}
    )
    for slot, indexer_name in enumerate(indexer_names):
        offset = indexer_base + slot * indexer_page * num_blocks
        add_tensor(indexer_name, attention_specs[indexer_name], offset)
        if tail_names:
            tail_name = tail_names[slot]
            add_tensor(tail_name, tail_specs[tail_name], offset)

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )


class Glm5NextKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Plan GLM-5.3's shared MLA, Mamba, indexer, and tail storage."""

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        if (
            vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            or vllm_config.attention_config.hisparse_config is not None
        ):
            return super().get_kv_cache_groups(vllm_config, kv_cache_spec)

        groups = _get_glm_groups(vllm_config, kv_cache_spec)
        if groups is not None:
            return groups
        return super().get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        layout = _get_glm_layout(kv_cache_groups)
        if layout is None:
            return super().get_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups, num_blocks
            )
        return _materialize_glm_config(vllm_config, kv_cache_groups, layout, num_blocks)
