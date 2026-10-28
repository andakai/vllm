# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.request import Request


@dataclass(frozen=True)
class BlockHashPolicy:
    """Shared bounds and event payloads for canonical request block hashes.

    Hash availability establishes identity, not KV readiness. Callers must
    supply a finalized prefix and retain per-group materialization checks.
    """

    hash_block_size: int = 1
    num_lookahead_tokens: int = 0

    def cacheable_tokens(
        self,
        block_hashes: Sequence["BlockHash"],
        num_finalized_tokens: int,
        alignment_tokens: int = 1,
    ) -> int:
        limit = min(num_finalized_tokens, len(block_hashes) * self.hash_block_size)
        return max(0, limit) // alignment_tokens * alignment_tokens

    def reusable_tokens(self, num_tokens: int, alignment_tokens: int = 1) -> int:
        # Leave a token to produce logits, as well as the hash dependency window.
        limit = max(0, num_tokens - max(1, self.num_lookahead_tokens))
        return limit // alignment_tokens * alignment_tokens

    def event_token_ids(self, request: "Request", start: int, end: int) -> list[int]:
        """Snapshot payload tokens together with their hash dependencies."""
        return list(request.all_token_ids[start : end + self.num_lookahead_tokens])

    def event_tokens(
        self,
        token_ids: Sequence[int] | None,
        start: int,
        end: int,
        block_size: int,
        token_ids_start: int = 0,
    ) -> tuple[list[int], list[list[int]] | None]:
        """Describe a stored range, or omit tokens when its snapshot is incomplete."""
        start -= token_ids_start
        end -= token_ids_start
        if (
            token_ids is None
            or start < 0
            or end + self.num_lookahead_tokens > len(token_ids)
        ):
            return [], None
        dependencies = (
            [
                list(token_ids[i : i + self.num_lookahead_tokens])
                for i in range(start + block_size, end + 1, block_size)
            ]
            if self.num_lookahead_tokens
            else None
        )
        return list(token_ids[start:end]), dependencies

    def event_extra_keys(
        self, request: "Request", start: int, end: int, start_mm_idx: int
    ) -> tuple[tuple[Any, ...] | None, int]:
        from vllm.v1.core.kv_cache_utils import (
            generate_block_hash_extra_keys,
            generate_lookahead_block_hash_extra_keys,
        )

        if self.num_lookahead_tokens:
            return generate_lookahead_block_hash_extra_keys(
                request, start, end, end + self.num_lookahead_tokens, start_mm_idx
            )
        return generate_block_hash_extra_keys(request, start, end, start_mm_idx)
