# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from typing import cast

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    create_kv_cache_group_specs,
)
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


def _pp_balanced_mamba_group_count(
    vllm_config: VllmConfig,
    mamba_layer_names: list[str],
    mla_layer_names: list[str],
) -> int | None:
    """Return a Mamba group count whose PP projections fit the MLA slots."""
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


def _get_kv_cache_groups_glm5_next(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build GLM-5.3-Flash groups with Mamba/MLA and tail/indexer aliasing."""
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, MambaSpec)
    }
    tail_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, KpoolTailSpec)
    }
    attn_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
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
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(attn_specs)
    assert uniform_spec is not None

    tail_group: KVCacheGroupSpec | None = None
    if tail_specs:
        idx_page = next(iter(idx_pages))
        padded_tail_specs: dict[str, KVCacheSpec] = {
            name: replace(spec, page_size_padded=idx_page)
            for name, spec in tail_specs.items()
        }
        tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
        assert tail_uniform is not None
        tail_group = KVCacheGroupSpec(list(padded_tail_specs), tail_uniform)

    any_mamba = next(iter(mamba_specs.values()))
    assert all(spec == any_mamba for spec in mamba_specs.values())
    if any_mamba.real_page_size_bytes > mla_page:
        raise ValueError(
            f"the mamba state page ({any_mamba.real_page_size_bytes} bytes) "
            f"does not fit the MLA page ({mla_page} bytes); increase tensor "
            "parallelism or use a wider KV cache dtype"
        )
    padded_specs: dict[str, KVCacheSpec] = {
        name: replace(any_mamba, page_size_padded=mla_page) for name in mamba_specs
    }
    num_groups = _pp_balanced_mamba_group_count(
        vllm_config, list(mamba_specs), mla_names
    )
    if num_groups is None:
        raise ValueError(
            "a pipeline stage has mamba layers but no MLA layer to share "
            "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
        )
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for index, name in enumerate(mamba_specs):
        mamba_grouped_names[index % num_groups].append(name)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )


def _glm5_next_tensor_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> (
    tuple[
        KVCacheGroupSpec,
        list[KVCacheGroupSpec],
        list[str],
        list[str],
        int,
        int,
        list[str],
        int,
    ]
    | None
):
    """Recognize the GLM-5.3-Flash grouping after optional PP projection."""
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

    attn_uniform = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
    mla_inner = cast(dict[str, MLAAttentionSpec], attn_uniform.kv_cache_specs)
    if not all(
        type(spec) is MLAAttentionSpec and spec.page_size_padded is None
        for spec in mla_inner.values()
    ):
        return None
    mla_names = [
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state == 1
    ]
    idx_names = [
        name for name in attn_group.layer_names if mla_inner[name].tokens_per_state > 1
    ]
    mla_pages = {mla_inner[name].page_size_bytes for name in mla_names}
    idx_pages = {mla_inner[name].page_size_bytes for name in idx_names}
    if len(mla_pages) != 1 or len(idx_pages) != 1:
        return None
    mla_page = mla_pages.pop()
    idx_page = idx_pages.pop()
    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):
        return None

    tail_names: list[str] = []
    tail_page = 0
    if tail_group is not None:
        tail_names = list(tail_group.layer_names)
        tail_inner = cast(
            UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
        ).kv_cache_specs
        tail_pages = {
            cast(KpoolTailSpec, spec).unpadded_page_size_bytes
            for spec in tail_inner.values()
        }
        if len(tail_pages) != 1 or len(tail_names) != len(idx_names):
            return None
        tail_page = tail_pages.pop()
        if tail_page > idx_page:
            return None

    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_page,
        tail_names,
        tail_page,
    )


class Glm5NextKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Plan GLM-5.3-Flash's shared Mamba, MLA, and indexer storage."""

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

        groups = _get_kv_cache_groups_glm5_next(vllm_config, kv_cache_spec)
        if groups is not None:
            return groups
        return super().get_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        if vllm_config.attention_config.hisparse_config is not None:
            return super().get_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups, available_memory
            )

        layout = _glm5_next_tensor_layout(kv_cache_groups)
        if layout is None:
            return super().get_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups, available_memory
            )

        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = layout
        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = self.may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        size = bytes_per_block * num_blocks
        attn_specs = cast(
            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
        ).kv_cache_specs
        tensors: list[KVCacheTensor] = []

        def add_tensor(layer_name: str, spec: KVCacheSpec, offset: int) -> None:
            tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=[layer_name],
                    layer_stride=spec.page_size_bytes * num_blocks,
                    block_stride=spec.page_size_bytes,
                    offset=offset,
                )
            )

        for index, mla_name in enumerate(mla_names):
            offset = index * mla_page * num_blocks
            add_tensor(mla_name, attn_specs[mla_name], offset)
            for group in mamba_groups:
                if index < len(group.layer_names):
                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)

        idx_base = len(mla_names) * mla_page * num_blocks
        for index, idx_name in enumerate(idx_names):
            offset = idx_base + index * idx_page * num_blocks
            add_tensor(idx_name, attn_specs[idx_name], offset)
            if tail_names:
                tail_name = tail_names[index]
                tail_group = next(
                    group for group in kv_cache_groups if tail_name in group.layer_names
                )
                tail_specs = cast(
                    UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
                ).kv_cache_specs
                add_tensor(tail_name, tail_specs[tail_name], offset)

        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    def _get_kv_cache_bytes_per_block(
        self, kv_cache_groups: list[KVCacheGroupSpec]
    ) -> int:
        layout = _glm5_next_tensor_layout(kv_cache_groups)
        if layout is None:
            return super()._get_kv_cache_bytes_per_block(kv_cache_groups)
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = layout
        return len(mla_names) * mla_page + len(idx_names) * idx_page

    def _max_memory_usage_bytes_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        layout = _glm5_next_tensor_layout(kv_cache_groups)
        if layout is None:
            return super()._max_memory_usage_bytes_from_groups(
                vllm_config, kv_cache_groups
            )
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _,
        ) = layout
        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
        total_blocks = uniform_spec.max_memory_usage_pages(vllm_config)
        total_blocks += sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in mamba_groups
        )
        if tail_names:
            total_blocks += 1
        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)
