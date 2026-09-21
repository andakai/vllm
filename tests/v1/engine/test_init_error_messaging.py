# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec

default_builder = DefaultKVCacheConfigBuilder()


def test_kv_cache_oom_no_memory():
    config = VllmConfig(model_config=ModelConfig(max_model_len=2048))
    config.cache_config.kv_cache_layout = "LBNHC"

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(ValueError):
        default_builder.get_kv_cache_configs(config, [spec], [0])


def test_kv_cache_oom_insufficient_memory(monkeypatch):
    config = VllmConfig(model_config=ModelConfig(max_model_len=2048))
    config.cache_config.kv_cache_layout = "LBNHC"

    monkeypatch.setattr(
        default_builder,
        "_get_max_memory_usage_bytes_from_groups",
        lambda c, g: 100 * 1024**3,  # 100 GiB
    )

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(ValueError):
        default_builder.get_kv_cache_configs(config, [spec], [1024**3])
