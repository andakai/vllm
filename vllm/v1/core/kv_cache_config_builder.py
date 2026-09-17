# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Template-method KV cache planning and per-config builder resolution."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        KVCacheSpec,
        KVCacheTensor,
    )

_DEFAULT_BUILDER = "vllm.v1.core.kv_cache_config_builder.KVCacheConfigBuilder"


def resolve_kv_cache_config_builder(
    vllm_config: VllmConfig,
) -> KVCacheConfigBuilder:
    """Resolve a fresh builder with platform > model > default priority."""
    from vllm.platforms import current_platform

    platform_builder = current_platform.get_kv_cache_config_builder_cls(vllm_config)
    model_builder = getattr(
        getattr(vllm_config, "model_config", None),
        "kv_cache_config_builder_cls",
        None,
    )
    qualname = next(
        value
        for value in (platform_builder, model_builder, _DEFAULT_BUILDER)
        if isinstance(value, str) and value
    )
    builder_cls = resolve_obj_by_qualname(qualname)
    if not isinstance(builder_cls, type) or not issubclass(
        builder_cls, KVCacheConfigBuilder
    ):
        raise TypeError(
            f"KV cache config builder {qualname!r} must subclass "
            f"{KVCacheConfigBuilder.__qualname__}"
        )
    return builder_cls()


class KVCacheConfigBuilder:
    """Own the common KV cache planning flow and expose four model hooks."""

    def get_kv_cache_configs(
        self,
        vllm_config: VllmConfig,
        kv_cache_specs: list[dict[str, KVCacheSpec]],
        available_memory: list[int],
    ) -> list[KVCacheConfig]:
        from vllm.v1.core.kv_cache_utils import _plan_kv_cache_configs

        return _plan_kv_cache_configs(
            self, vllm_config, kv_cache_specs, available_memory
        )

    def get_profiling_kv_cache_config(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
        min_blocks: int,
    ) -> KVCacheConfig:
        """Build the minimal cache through the same hooks as final planning."""
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        KVCacheSpecRegistry.check_kv_cache_spec_registry(kv_cache_spec)
        groups = self.get_kv_cache_groups(vllm_config, kv_cache_spec)
        saved_override = vllm_config.cache_config.num_gpu_blocks_override
        vllm_config.cache_config.num_gpu_blocks_override = min_blocks
        try:
            return self.get_kv_cache_config_from_groups(
                vllm_config, groups, available_memory=0
            )
        finally:
            vllm_config.cache_config.num_gpu_blocks_override = saved_override

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        from vllm.v1.core.kv_cache_utils import _get_default_kv_cache_groups

        if (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and vllm_config.attention_config.hisparse_config is None
            and kv_cache_spec
        ):
            groups = self._get_custom_kv_cache_groups(vllm_config, kv_cache_spec)
            if groups is not None:
                return groups
        return _get_default_kv_cache_groups(vllm_config, kv_cache_spec)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> KVCacheConfig:
        from vllm.v1.hisparse.layout import (
            get_hisparse_host_pool_bytes,
            get_hisparse_kv_cache_config,
        )

        if not kv_cache_groups:
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
                prefix_cache_retention_interval=(
                    vllm_config.cache_config.prefix_cache_retention_interval
                ),
            )
        if vllm_config.attention_config.hisparse_config is not None:
            return get_hisparse_kv_cache_config(
                vllm_config,
                kv_cache_groups,
                available_memory,
                get_hisparse_host_pool_bytes(vllm_config),
            )

        bytes_per_block = self.get_pool_bytes_per_block(kv_cache_groups)
        num_blocks = self.may_override_num_blocks(
            vllm_config, available_memory // bytes_per_block
        )
        size = bytes_per_block * num_blocks
        tensors = self._build_kv_cache_tensors(
            vllm_config,
            kv_cache_groups,
            num_blocks,
            size,
            bytes_per_block,
        )
        expected_layers = sorted(
            name for group in kv_cache_groups for name in group.layer_names
        )
        actual_layers = sorted(name for tensor in tensors for name in tensor.layers)
        assert actual_layers == expected_layers, (
            "KV cache tensor descriptors must cover each grouped layer exactly once"
        )
        assert all(tensor.size == size for tensor in tensors), (
            "KV cache tensor descriptors must share the core-sized allocation"
        )
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    def get_pool_bytes_per_block(self, kv_cache_groups: list[KVCacheGroupSpec]) -> int:
        return self._get_kv_cache_bytes_per_block(kv_cache_groups)

    def get_max_memory_usage_bytes(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        return self._get_max_memory_usage_bytes(vllm_config, kv_cache_groups)

    def check_enough_kv_cache_memory(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
        available_memory: int,
    ) -> None:
        from vllm.v1.core.kv_cache_utils import _check_enough_kv_cache_memory

        if not kv_cache_spec:
            return
        groups = self.get_kv_cache_groups(vllm_config, dict(kv_cache_spec))
        check_memory = available_memory - self.get_pool_bytes_per_block(groups)
        _check_enough_kv_cache_memory(
            check_memory,
            partial(self.get_max_memory_usage_bytes, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(self._estimate_max_model_len, vllm_config, groups),
        )

    @staticmethod
    def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
        override = vllm_config.cache_config.num_gpu_blocks_override
        return override if override is not None else num_blocks

    def _estimate_max_model_len(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        available_memory: int,
    ) -> int:
        original_max = vllm_config.model_config.max_model_len
        hisparse_enabled = (
            vllm_config.attention_config.hisparse_config is not None
            and bool(kv_cache_groups)
        )

        def fits(model_len: int) -> bool:
            vllm_config.model_config.max_model_len = model_len
            if hisparse_enabled:
                from vllm.v1.core.kv_cache_utils import (
                    get_max_concurrency_for_kv_cache_config,
                )

                try:
                    config = self.get_kv_cache_config_from_groups(
                        vllm_config, kv_cache_groups, available_memory
                    )
                except ValueError:
                    return False
                return get_max_concurrency_for_kv_cache_config(vllm_config, config) >= 1
            return (
                self.get_max_memory_usage_bytes(vllm_config, kv_cache_groups)
                <= available_memory
            )

        try:
            left, right = 1, original_max
            if not fits(left):
                return 0
            result = 1
            while left <= right:
                mid = (left + right) // 2
                if fits(mid):
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1
            return result
        finally:
            vllm_config.model_config.max_model_len = original_max

    def _auto_fit_max_model_len(
        self,
        vllm_config: VllmConfig,
        projected_groups_per_worker: list[list[KVCacheGroupSpec]],
        available_memory: list[int],
    ) -> None:
        from vllm.utils.mem_utils import format_gib
        from vllm.v1.core.kv_cache_utils import logger

        original_max = vllm_config.model_config.max_model_len
        if all(not groups for groups in projected_groups_per_worker):
            logger.info_once(
                "Auto-fit max_model_len: attention-free model, "
                "using derived max_model_len=%d",
                original_max,
            )
            return

        auto_fit_max = original_max
        limiting_worker_mem = available_memory[0]
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                continue
            worker_max = self._estimate_max_model_len(vllm_config, groups, avail_mem)
            if worker_max < auto_fit_max:
                auto_fit_max = worker_max
                limiting_worker_mem = avail_mem

        if auto_fit_max <= 0:
            raise ValueError(
                "Cannot auto-fit max_model_len: not enough GPU memory available "
                "to serve even a single token. Try increasing "
                "`gpu_memory_utilization`."
            )
        if auto_fit_max >= original_max:
            logger.info_once(
                "Auto-fit max_model_len: full model context length %d fits in "
                "available GPU memory",
                original_max,
            )
            return
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
        )

    def _get_custom_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec] | None:
        return None

    def _build_kv_cache_tensors(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
        size: int,
        bytes_per_block: int,
    ) -> list[KVCacheTensor]:
        from vllm.v1.core.kv_cache_utils import _build_default_kv_cache_tensors

        return _build_default_kv_cache_tensors(
            vllm_config,
            kv_cache_groups,
            num_blocks,
            size,
            bytes_per_block,
        )

    def _get_kv_cache_bytes_per_block(
        self, kv_cache_groups: list[KVCacheGroupSpec]
    ) -> int:
        from vllm.v1.core.kv_cache_utils import _get_kv_cache_bytes_per_block

        return _get_kv_cache_bytes_per_block(kv_cache_groups)

    def _get_max_memory_usage_bytes(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> int:
        from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups

        return _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
