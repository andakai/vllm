# Disk Engine

The disk weight transfer engine loads safetensors checkpoints from storage shared by the trainer and every inference worker. Each `update_weights` request names a checkpoint directory (or a single `.safetensors` file), and every worker reads it with the same safetensors iterator as a cold start. The trainer needs no NCCL path to the server's GPUs, and the request carries only the path.

## When to Use Disk

- The trainer already saves checkpoints (for example with `save_pretrained`) to a filesystem the inference nodes mount, and has no NCCL path to them.
- Asynchronous or disaggregated RL, where trainer and rollout clusters only share storage.

For full models on reachable GPUs, NCCL is faster: disk transfer writes and reads every byte through storage.

## Inference Side

```bash
vllm serve my-model --weight-transfer-config '{"backend": "disk"}'
```

`init_transfer_engine` sets nothing up. `start_weight_update` and `finish_weight_update` run the same reload as the IPC and NCCL engines (see [direct reload](../direct_reload.md) and [layerwise reload](../layerwise.md)), so the checkpoint uses the checkpoint's own parameter names, as at startup.

To load a checkpoint the trainer saved itself, call the routes directly:

```bash
curl -X POST http://server:8000/init_weight_transfer_engine -d '{"init_info": {}}' -H 'Content-Type: application/json'
curl -X POST http://server:8000/start_weight_update
curl -X POST http://server:8000/update_weights -d '{"update_info": {"path": "/shared/ckpt/step_100"}}' -H 'Content-Type: application/json'
curl -X POST http://server:8000/finish_weight_update -d '{"weight_version": "step_100"}' -H 'Content-Type: application/json'
```

In a directory with a `model.safetensors.index.json`, only the indexed files are loaded, which skips extra files such as `consolidated.safetensors`.

## Trainer Side

The trainer engine writes the source's tensors as safetensors shards in a fresh subdirectory of `directory`, has the server load them, and removes them:

```python
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.disk_engine import DiskTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=DiskTrainerInitInfo(rank=0, directory="/shared/weight-updates"),
    client=HTTPVLLMWeightSyncClient("http://server:8000"),
    source=ModuleSource(train_model),
)
engine.send_weights()  # write shards, start, update_weights(path), finish, remove
```

`shard_size_bytes` (default 4 GiB) bounds the tensor bytes per file; a larger tensor gets its own file. With several trainer ranks, every rank iterates the source (which may run collectives), and only rank 0, which must own every parameter, writes.

Every worker reads every file: tensor-parallel workers each load the full tensors and keep their shards, as at startup.
