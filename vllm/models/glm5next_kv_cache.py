# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from typing import NamedTuple, cast

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    KVCachePlanningPolicy,
    create_kv_cache_group_specs,
)
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)


class _Glm5NextLayout(NamedTuple):
    attn_group: KVCacheGroupSpec
    mamba_groups: list[KVCacheGroupSpec]
    mla_names: list[str]
    indexer_names: list[str]
    mla_page: int
    indexer_page: int
    tail_names: list[str]


class Glm5NextKVCachePlanningPolicy(KVCachePlanningPolicy):
    """GLM-5.3-Flash cache grouping and shared tensor placement."""

    def _get_mamba_group_count(
        self, mamba_names: list[str], mla_names: list[str]
    ) -> int | None:
        num_groups = cdiv(len(mamba_names), len(mla_names))
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        if pp_size == 1:
            return num_groups

        from vllm.distributed.utils import get_pp_indices
        from vllm.model_executor.models.utils import extract_layer_index

        total_layers = self.vllm_config.model_config.get_total_num_hidden_layers()
        mamba_indices = [extract_layer_index(name) for name in mamba_names]
        mla_indices = [extract_layer_index(name) for name in mla_names]
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

    def get_kv_cache_groups(
        self, kv_cache_spec: dict[str, KVCacheSpec]
    ) -> list[KVCacheGroupSpec] | None:
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
        indexer_pages = {
            spec.page_size_bytes
            for spec in mla_specs.values()
            if spec.tokens_per_state > 1
        }
        if not indexer_pages:
            return None

        assert all(spec.page_size_padded is None for spec in mla_specs.values())
        assert len(indexer_pages) == 1
        mla_names = [
            name for name, spec in mla_specs.items() if spec.tokens_per_state == 1
        ]
        mla_pages = {mla_specs[name].page_size_bytes for name in mla_names}
        assert len(mla_pages) == 1
        mla_page = mla_pages.pop()
        attn_uniform = UniformTypeKVCacheSpecs.from_specs(attn_specs)
        assert attn_uniform is not None

        tail_group: KVCacheGroupSpec | None = None
        if tail_specs:
            indexer_page = next(iter(indexer_pages))
            padded_tail_specs: dict[str, KVCacheSpec] = {
                name: replace(spec, page_size_padded=indexer_page)
                for name, spec in tail_specs.items()
            }
            tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
            assert tail_uniform is not None
            tail_group = KVCacheGroupSpec(list(padded_tail_specs), tail_uniform)

        mamba_spec = next(iter(mamba_specs.values()))
        assert all(spec == mamba_spec for spec in mamba_specs.values())
        if mamba_spec.real_page_size_bytes > mla_page:
            raise ValueError(
                f"the mamba state page ({mamba_spec.real_page_size_bytes} bytes) "
                f"does not fit the MLA page ({mla_page} bytes); increase tensor "
                "parallelism or use a wider KV cache dtype"
            )
        padded_mamba_specs: dict[str, KVCacheSpec] = {
            name: replace(mamba_spec, page_size_padded=mla_page) for name in mamba_specs
        }
        num_groups = self._get_mamba_group_count(list(mamba_specs), mla_names)
        if num_groups is None:
            raise ValueError(
                "a pipeline stage has mamba layers but no MLA layer to share "
                "slots with; realign the stage boundaries "
                "(VLLM_PP_LAYER_PARTITION)"
            )
        grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
        for index, name in enumerate(mamba_specs):
            grouped_names[index % num_groups].append(name)

        return (
            [KVCacheGroupSpec(list(attn_specs), attn_uniform)]
            + ([tail_group] if tail_group is not None else [])
            + create_kv_cache_group_specs(padded_mamba_specs, grouped_names)
        )

    @staticmethod
    def _get_layout(
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> _Glm5NextLayout | None:
        uniform_groups = [
            group
            for group in kv_cache_groups
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        ]
        mamba_groups = [
            group
            for group in kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
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

        attn_specs = cast(
            UniformTypeKVCacheSpecs, attn_group.kv_cache_spec
        ).kv_cache_specs
        if not all(
            type(spec) is MLAAttentionSpec and spec.page_size_padded is None
            for spec in attn_specs.values()
        ):
            return None
        mla_names = [
            name
            for name in attn_group.layer_names
            if cast(MLAAttentionSpec, attn_specs[name]).tokens_per_state == 1
        ]
        indexer_names = [
            name
            for name in attn_group.layer_names
            if cast(MLAAttentionSpec, attn_specs[name]).tokens_per_state > 1
        ]
        mla_pages = {attn_specs[name].page_size_bytes for name in mla_names}
        indexer_pages = {attn_specs[name].page_size_bytes for name in indexer_names}
        if len(mla_pages) != 1 or len(indexer_pages) != 1:
            return None
        mla_page = mla_pages.pop()
        indexer_page = indexer_pages.pop()
        if any(
            group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups
        ):
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
                or len(tail_names) != len(indexer_names)
                or tail_pages.pop() > indexer_page
            ):
                return None
        return _Glm5NextLayout(
            attn_group,
            mamba_groups,
            mla_names,
            indexer_names,
            mla_page,
            indexer_page,
            tail_names,
        )

    def get_kv_cache_bytes_per_block(
        self, kv_cache_groups: list[KVCacheGroupSpec]
    ) -> int | None:
        if (layout := self._get_layout(kv_cache_groups)) is None:
            return None
        return (
            len(layout.mla_names) * layout.mla_page
            + len(layout.indexer_names) * layout.indexer_page
        )

    def get_kv_cache_tensors(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
        bytes_per_block: int,
    ) -> list[KVCacheTensor] | None:
        if (layout := self._get_layout(kv_cache_groups)) is None:
            return None
        size = bytes_per_block * num_blocks
        attn_specs = cast(
            UniformTypeKVCacheSpecs, layout.attn_group.kv_cache_spec
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

        for index, mla_name in enumerate(layout.mla_names):
            offset = index * layout.mla_page * num_blocks
            add_tensor(mla_name, attn_specs[mla_name], offset)
            for group in layout.mamba_groups:
                if index < len(group.layer_names):
                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)

        indexer_base = len(layout.mla_names) * layout.mla_page * num_blocks
        for index, indexer_name in enumerate(layout.indexer_names):
            offset = indexer_base + index * layout.indexer_page * num_blocks
            add_tensor(indexer_name, attn_specs[indexer_name], offset)
            if layout.tail_names:
                tail_name = layout.tail_names[index]
                tail_group = next(
                    group for group in kv_cache_groups if tail_name in group.layer_names
                )
                tail_specs = cast(
                    UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
                ).kv_cache_specs
                add_tensor(tail_name, tail_specs[tail_name], offset)
        return tensors

    def get_max_memory_usage_bytes(
        self, kv_cache_groups: list[KVCacheGroupSpec]
    ) -> int | None:
        if (layout := self._get_layout(kv_cache_groups)) is None:
            return None
        attn_spec = cast(UniformTypeKVCacheSpecs, layout.attn_group.kv_cache_spec)
        total_blocks = attn_spec.max_memory_usage_pages(self.vllm_config)
        total_blocks += sum(
            cdiv(
                group.kv_cache_spec.max_memory_usage_bytes(self.vllm_config),
                group.kv_cache_spec.page_size_bytes,
            )
            for group in layout.mamba_groups
        )
        if layout.tail_names:
            total_blocks += 1
        bytes_per_block = self.get_kv_cache_bytes_per_block(kv_cache_groups)
        assert bytes_per_block is not None
        return total_blocks * bytes_per_block
