# Inline Engine

The inline weight transfer engine carries the tensors in the `update_weights` requests themselves. The trainer serializes each chunk with [safetensors](https://github.com/huggingface/safetensors) and base64-encodes it, and every worker decodes the chunk and loads it with `load_weights`. Only the control plane connects the trainer and the server (the HTTP routes or Ray), so the transfer needs no NCCL rendezvous, no shared filesystem and no pickling.

## When to Use Inline

- The trainer can reach the server's HTTP port but not its GPUs: no NCCL path between the two clusters, and no shared filesystem.
- Small updates such as draft models or LoRA adapters, where setting up a data plane is not worth it.

For full models on reachable GPUs, NCCL is much faster: inline moves every byte through the CPU, base64 and the HTTP server.

## Inference Side

```bash
vllm serve my-model --weight-transfer-config '{"backend": "inline"}'
```

`init_transfer_engine` sets nothing up. `start_weight_update` and `finish_weight_update` run the same layerwise reload as the IPC engine, so the trainer sends checkpoint-format weights.

## Trainer Side

```python
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.inline_engine import InlineTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=InlineTrainerInitInfo(rank=0),  # chunk_size_bytes=16 MiB
    client=HTTPVLLMWeightSyncClient("http://server:8000"),
    source=ModuleSource(train_model),
)
engine.send_weights()  # start, one update_weights per chunk, finish
```

`chunk_size_bytes` bounds the tensor bytes per request; a larger tensor is sent alone. Base64 adds a third, so the default 16 MiB chunk makes requests of about 21 MiB, within the Rust frontend's default 32 MiB request body limit. A tensor is never split, so a tensor over 24 MiB, such as most embedding tables, needs a larger limit with the Rust frontend: set `VLLM_HTTP_MAX_REQUEST_BODY_SIZE` on the server (`0` removes the limit; the Python frontend has none). With several trainer ranks, every rank iterates the source (which may run collectives), and only rank 0, which must own every parameter, sends.

Every worker receives every chunk, as with the IPC engine: tensor-parallel workers each load the full tensors and keep their shards.
