# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interface and resolution for pluggable KV cache config builders."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )


class KVCacheConfigBuilder(ABC):
    """Interface for model- or platform-specific KV cache planning.

    Subclasses normally inherit from ``DefaultKVCacheConfigBuilder`` and
    override only the hooks they need. A platform may replace the top-level
    planning method when it cannot use Core's cross-worker planning flow.
    """

    @abstractmethod
    def get_kv_cache_configs(
        self,
        vllm_config: "VllmConfig",
        kv_cache_specs: list[dict[str, "KVCacheSpec"]],
        available_memory: list[int],
    ) -> list["KVCacheConfig"]:
        """Generate the full KV cache configurations for every worker.

        The main entry point: takes the per-worker KV cache specs and the
        memory available on each worker, runs the whole planning pipeline
        (merge specs, group layers, project to workers, auto-fit
        max_model_len, admission checks, per-worker layouts, min-blocks
        convergence), and returns one ready-to-allocate
        :class:`~vllm.v1.kv_cache_interface.KVCacheConfig` per worker.

        Consumed by ``vllm/v1/engine/core.py``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_spec: dict[str, "KVCacheSpec"],
    ) -> list["KVCacheGroupSpec"]:
        """Organize layer specs into scheduler-visible cache groups."""
        raise NotImplementedError

    @abstractmethod
    def get_pool_bytes_per_block(
        self, kv_cache_groups: list["KVCacheGroupSpec"]
    ) -> int:
        """Return bytes consumed by one global block ID in the physical pool."""
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_config_from_groups(
        self,
        vllm_config: "VllmConfig",
        kv_cache_groups: list["KVCacheGroupSpec"],
        num_blocks: int,
    ) -> "KVCacheConfig":
        """Materialize groups for exactly ``num_blocks`` global block IDs."""
        raise NotImplementedError


_active_builder: KVCacheConfigBuilder | None = None


def get_kv_cache_config_builder(
    vllm_config: "VllmConfig",
) -> KVCacheConfigBuilder:
    """Resolve and cache the builder selected by the current platform."""
    global _active_builder
    if _active_builder is None:
        from vllm.platforms import current_platform

        builder_cls = resolve_obj_by_qualname(
            current_platform.get_kv_cache_config_builder_cls(vllm_config)
        )
        _active_builder = builder_cls()
    return _active_builder


def _get_profiling_kv_cache_config(
    vllm_config: "VllmConfig",
    kv_cache_spec: dict[str, "KVCacheSpec"],
    min_blocks: int,
) -> "KVCacheConfig":
    """Build profiling storage through the active builder's normal hooks."""
    builder = get_kv_cache_config_builder(vllm_config)
    groups = builder.get_kv_cache_groups(vllm_config, kv_cache_spec)
    return builder.get_kv_cache_config_from_groups(
        vllm_config, groups, max(min_blocks, 1)
    )
