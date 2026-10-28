# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External-store cache-hit coordinator for MooncakeStoreConnector."""

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, NamedTuple, cast

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    chunk_hashes_for_block_size,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    resolve_dcp_kv_cache_spec,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    get_reusable_prefix_length,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class StoreSpecGroup(NamedTuple):
    spec: KVCacheSpec
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by a ``(group_id, hash)`` exists set."""

    def __init__(
        self,
        hash_block_size: int,
        exists: set[tuple[int, bytes]] | None = None,
    ) -> None:
        # ``exists=None`` is used on the recv side where hit_length is already
        # determined and we just want each spec's manager to apply its own mask.
        self._exists = exists
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        # Dummy ID 1 for present block for duck-typing.
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        # Mirrors BlockPool.get_cached_block: hit only when every group_id
        # (groups sharing a spec) has the hash cached.
        if self._exists is None:
            return [self._present_block] * len(group_ids)
        h = bytes(block_hash)
        if all((g, h) in self._exists for g in group_ids):
            return [self._present_block] * len(group_ids)
        return None

    def contains(self, group_id: int, block_hash: BlockHash) -> bool:
        """Return whether a group has a loadable key for this hash."""
        return (
            self._exists is not None and (group_id, bytes(block_hash)) in self._exists
        )


class MooncakeStoreCoordinator:
    """Mirror of ``HybridKVCacheCoordinator.find_longest_cache_hit`` over an
    ``ExternalCachedBlockPool``."""

    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        retention_interval: int | None = None,
        enable_partial_hash_hits: bool = False,
        num_hash_lookahead_tokens: int = 0,
    ) -> None:
        # Mirrors core's resolve_kv_cache_block_sizes: the hash unit only has
        # to divide groups that participate in prefix caching. Non-shareable
        # scratch groups (e.g. GLM-5.3-Flash's kpool tail) are skipped by
        # _verify_and_split_kv_cache_groups and never probed for hits.
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_groups
            if g.kv_cache_spec.prefix_cacheable
        ), "block_size must be divisible by hash_block_size"
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of "
            f"hash_block_size ({hash_block_size})"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_groups
            if g.kv_cache_spec.prefix_cacheable
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self.mamba_group_ids = {
            group_id
            for group_id, group in enumerate(kv_cache_groups)
            if isinstance(_unwrap_spec(group.kv_cache_spec), MambaSpec)
        }
        self.num_hash_lookahead_tokens = num_hash_lookahead_tokens
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        # Core's decision: whether hits land on the hash unit, below the block.
        self.enable_partial_hash_hits = enable_partial_hash_hits
        # Mirror vLLM core's KVCacheCoordinator.retention_interval.
        self.retention_interval = retention_interval
        self._verify_and_split_kv_cache_groups()

    @classmethod
    def from_kv_cache_config(
        cls,
        kv_cache_config: KVCacheConfig,
        vllm_config: "VllmConfig",
        scheduler_block_size: int,
        hash_block_size: int,
    ) -> "MooncakeStoreCoordinator":
        """Build the coordinator the connector's scheduler and workers share.

        Specs are DCP-resolved, so a block spans the tokens of one block-table
        entry. Partial hits follow core's hit alignment, which checks every KV
        cache group and its manager, not only the groups the store transfers;
        unset (a config the engine core did not resolve) keeps hits on the
        block.
        """
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        alignment = kv_cache_config.cache_hit_alignment_tokens
        return cls(
            [
                replace(
                    group,
                    kv_cache_spec=resolve_dcp_kv_cache_spec(
                        group.kv_cache_spec, dcp_world_size
                    ),
                )
                for group in kv_cache_config.prefix_cacheable_groups
            ],
            scheduler_block_size,
            hash_block_size,
            retention_interval=kv_cache_config.prefix_cache_retention_interval,
            num_hash_lookahead_tokens=vllm_config.num_hash_lookahead_tokens,
            enable_partial_hash_hits=(
                alignment is not None and alignment < scheduler_block_size
            ),
        )

    def align_lookup_length(self, length: int) -> int:
        alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        return length // alignment * alignment

    def _verify_and_split_kv_cache_groups(self) -> None:
        """Mirrors KVCacheCoordinator.verify_and_split_kv_cache_groups but
        dispatches via spec_manager_map (we don't allocate managers).
        """
        attention_groups: list[StoreSpecGroup] = []
        for i, g in enumerate(self.kv_cache_groups):
            # Skip groups that opt out of prefix caching (e.g. GLM-5.3-Flash
            # kpool tail): per-request scratch, never shareable, so they must
            # not participate in hit lookup. Mirrors core's
            # KVCacheCoordinator.verify_and_split_kv_cache_groups.
            if not g.kv_cache_spec.prefix_cacheable:
                continue
            spec = _coverage_spec(g.kv_cache_spec)
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None, (
                f"No manager registered for KVCacheSpec {spec}"
            )
            for group in attention_groups:
                if group.spec == spec:
                    assert manager_cls is group.manager_cls
                    group.group_ids.append(i)
                    break
            else:
                attention_groups.append(StoreSpecGroup(spec, [i], manager_cls))
        # Full attention first (matches upstream convergence ordering).
        attention_groups.sort(key=lambda g: not isinstance(g.spec, FullAttentionSpec))
        self.attention_groups = attention_groups

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> tuple[tuple[list[bool], ...], int]:
        """Returns ``(load_mask_per_group, hit_length)``. ``mask[g][i]`` is True iff
        group ``g`` populates chunk ``i`` locally (e.g. SWA and Mamba tail-only);
        recv-side callers skip False slots.
        """
        blocks_per_group, hit_length = self._find_hit_blocks(
            block_hashes,
            max_length,
            cached_block_pool,
        )
        masks = tuple(
            [blk is not cached_block_pool.null_block for blk in blocks]
            for blocks in blocks_per_group
        )
        return masks, hit_length

    def load_mask(
        self,
        block_hashes: Sequence[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        """Per-group load masks: ``mask[g][i]`` is True iff group ``g``'s
        spec would populate chunk ``i`` locally at length ``token_len``
        (e.g. SWA / Mamba tail-only).
        """
        masks, _ = self.find_longest_cache_hit(
            block_hashes,
            token_len,
            ExternalCachedBlockPool(self.hash_block_size),
        )
        return masks

    def store_mask(
        self,
        aligned_token_len: int,
        start_token: int = 0,
        num_prompt_tokens: int | None = None,
    ) -> tuple[list[bool] | None, ...]:
        """Per-group store masks for the suffix starting at ``start_token``.

        ``mask[g][i]`` is True iff the i-th chunk of group ``g`` *after*
        ``start_token`` should be written to the store so a future cache hit
        can consume it. ``None`` is the all-True sentinel for the suffix.

        Reuses the engine's ``SingleTypeKVCacheManager.reachable_block_mask``
        so the store retains exactly the blocks the local prefix cache would.

        Mamba groups are always all-False: the normal save resolves blocks
        positionally from the connector's append-only block-ID snapshot, but
        an align-mode mamba block table is not append-only (interior state
        blocks are nulled/freed, and speculative decoding relocates the spec
        blocks in place), so a positional read may hit a null, freed, or live
        speculative-state block and persist wrong bytes under a valid prefix
        hash. Mamba state is persisted only through the connector-pinned exact
        block hand-off path
        (``SchedulerOutput.kv_connector_block_state.boundary_state_offloads``).
        """
        return self._reachable_masks(
            aligned_token_len,
            start_token,
            retention_interval=self.retention_interval,
            num_prompt_tokens=num_prompt_tokens,
            exclude_mamba=True,
        )

    def lookup_mask(
        self,
        aligned_token_len: int,
    ) -> tuple[list[bool] | None, ...]:
        """Per-group lookup masks.

        ``mask[g][i]`` is True iff chunk ``i`` of group ``g`` should be
        looked up as an aligned hit boundary. ``None`` is the all-True
        sentinel.
        """
        return self._reachable_masks(
            aligned_token_len,
            0,
            retention_interval=None,
            num_prompt_tokens=None,
        )

    def _reachable_masks(
        self,
        aligned_token_len: int,
        start_token: int,
        *,
        retention_interval: int | None,
        num_prompt_tokens: int | None,
        exclude_mamba: bool = False,
    ) -> tuple[list[bool] | None, ...]:
        mask_alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        assert aligned_token_len % mask_alignment == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"{mask_alignment}"
        )
        reachable_boundaries = (
            ()
            if num_prompt_tokens is None
            else (
                get_reusable_prefix_length(
                    num_prompt_tokens,
                    num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
                ),
            )
        )
        masks: list[list[bool] | None] = []
        for g_idx, g in enumerate(self.kv_cache_groups):
            spec = _coverage_spec(g.kv_cache_spec)
            end_chunk = aligned_token_len // spec.block_size
            start_chunk = min(end_chunk, max(0, cdiv(start_token, spec.block_size)))
            if exclude_mamba and isinstance(spec, MambaSpec):
                masks.append([False] * (end_chunk - start_chunk))
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None
            mask = manager_cls.reachable_block_mask(
                start_block=start_chunk,
                end_block=end_chunk,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                retention_interval=retention_interval,
                reachable_boundaries=reachable_boundaries,
                # ``spec`` is already DCP-resolved (worker.py applies
                # resolve_dcp_kv_cache_spec) and ``end_chunk`` is indexed in
                # that scaled block size, so the mask must not scale again.
                dcp_world_size=1,
            )
            if mask is not None:
                assert len(mask) == end_chunk - start_chunk
            masks.append(mask)
        return tuple(masks)

    def block_hashes_for_spec(
        self, block_hashes: Sequence[BlockHash], spec: KVCacheSpec
    ) -> Sequence[BlockHash]:
        return chunk_hashes_for_block_size(
            block_hashes, self.hash_block_size, spec.block_size
        )

    def _find_hit_blocks(
        self,
        block_hashes: Sequence[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Mirrors HybridKVCacheCoordinator.find_longest_cache_hit but
        dispatches via the registry (we don't allocate managers).
        """
        alignment_tokens = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        if len(self.attention_groups) == 1:
            spec, group_ids, manager_cls = self.attention_groups[0]
            hit_blocks, hit_length = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,  # type: ignore[arg-type]
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                alignment_tokens=alignment_tokens,
            )
            num_groups = len(self.kv_cache_groups)
            blocks_by_group: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
            for gid, blks in zip(group_ids, hit_blocks, strict=True):
                blocks_by_group[gid] = blks
            return tuple(blocks_by_group), hit_length

        num_groups = len(self.kv_cache_groups)
        hit_length = max_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        while True:
            curr_hit_length = hit_length

            for spec, group_ids, manager_cls in self.attention_groups:
                first_group_id = group_ids[0]
                cached = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    continue
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,  # type: ignore[arg-type]
                    max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    alignment_tokens=alignment_tokens,
                )
                curr_hit_length = _new_hit_length
                for gid, blocks in zip(group_ids, hit_blocks, strict=True):
                    hit_blocks_by_group[gid] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full-attention hit_blocks to final converged length;
        # other specs already trim themselves inside their hit logic. cdiv keeps
        # the partial tail block when hit_length is not block-aligned.
        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            num_blocks = cdiv(hit_length, group.spec.block_size)
            for group_id in group.group_ids:
                full_blks = hit_blocks_by_group[group_id]
                assert full_blks is not None
                del full_blks[num_blocks:]

        return (
            tuple(blks if blks is not None else [] for blks in hit_blocks_by_group),
            hit_length,
        )


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def _coverage_spec(spec: KVCacheSpec) -> KVCacheSpec:
    """The spec whose window covers every block an external hit restores.

    An external SWA hit allocates the MTP re-prefill tail
    (``extra_retained_tokens``) below the window, so lookup, load and store
    must cover it too, as the offloading connector does.
    """
    spec = _unwrap_spec(spec)
    if isinstance(spec, SlidingWindowSpec) and spec.extra_retained_tokens:
        return replace(
            spec,
            sliding_window=spec.sliding_window + spec.extra_retained_tokens,
            extra_retained_tokens=0,
        )
    return spec
