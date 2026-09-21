# KV cache config builders

KV cache config builders let a model or platform customize cache planning
without copying the engine-wide planning flow. The extension contract has one
entry point and three hooks:

| Method | Owner | Contract |
| --- | --- | --- |
| `get_kv_cache_configs` | Core | Plan every worker, apply global invariants, and return final configs. A platform may override this only when it must replace the complete flow. |
| `get_kv_cache_groups` | Model or platform | Convert layer specs into scheduler-visible groups. |
| `get_pool_bytes_per_block` | Model or platform | Report the physical pool cost of one global block ID. |
| `get_kv_cache_config_from_groups` | Model or platform | Materialize groups for an exact `num_blocks`, including backing sizes, aliases, offsets, and strides. |

Core owns spec validation, MTP retention, pipeline-stage projection,
`num_gpu_blocks_override`, null-block reservation, automatic model-length
fitting, admission checks, and cross-rank block-count convergence. These are
not customization hooks.

Profiling also uses `get_kv_cache_groups` and
`get_kv_cache_config_from_groups`. It passes the required minimum block count
directly, so a custom placement has one materialization implementation for
both profiling and final allocation.

## Model registration

A model declares its builder by qualified name:

```python
class MyModel(nn.Module):
    kv_cache_config_builder_cls = "my_package.MyKVCacheConfigBuilder"
```

The default platform returns this declaration when present. A platform may
return its own builder instead, or explicitly delegate to the model's choice.

## GLM-5.3-Flash example

`Glm5NextKVCacheConfigBuilder` demonstrates all three hooks:

```python
class Glm5NextKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
        groups = _get_glm_groups(vllm_config, kv_cache_spec)
        if groups is None:
            return super().get_kv_cache_groups(vllm_config, kv_cache_spec)
        return groups

    def get_pool_bytes_per_block(self, kv_cache_groups):
        layout = _get_glm_layout(kv_cache_groups)
        if layout is None:
            return super().get_pool_bytes_per_block(kv_cache_groups)
        return _get_glm_pool_bytes_per_block(layout)

    def get_kv_cache_config_from_groups(
        self, vllm_config, kv_cache_groups, num_blocks
    ):
        layout = _get_glm_layout(kv_cache_groups)
        if layout is None:
            return super().get_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups, num_blocks
            )
        return _materialize_glm_config(
            vllm_config, kv_cache_groups, layout, num_blocks
        )
```

The model-owned helpers implement GLM-specific behavior:

- `_get_glm_groups` creates PP-safe MLA/Mamba groups and pads Mamba and tail
  pages to the physical slots they share.
- `_get_glm_layout` recognizes the projected groups on each worker.
- `_get_glm_pool_bytes_per_block` charges only the MLA and indexer slots that
  own physical storage.
- `_materialize_glm_config` aliases Mamba views onto MLA slots and tail views
  onto indexer slots by assigning matching offsets and strides.

The complete implementation is in
`vllm/models/glm5next/kv_cache_config.py`. Core does not import GLM types or
recognize GLM model names.

## Choosing the smallest override

- Override only `get_kv_cache_groups` when the default physical packing is
  correct after custom grouping.
- Also override `get_pool_bytes_per_block` and
  `get_kv_cache_config_from_groups` when aliasing or placement changes the
  physical footprint.
- Override `get_kv_cache_configs` only for a platform whose capacity model or
  cross-worker orchestration cannot use the Core flow. Such a platform must
  still implement the three hooks consistently because profiling uses them.
