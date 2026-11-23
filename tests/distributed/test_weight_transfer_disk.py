# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The disk engine round trip: the trainer writes the source's tensors as
safetensors shards, the worker engine loads them, and the shards are removed."""

import json
import os
from functools import cached_property

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.disk_engine import (
    DiskTrainerInitInfo,
    DiskTrainerWeightTransferEngine,
    DiskWeightTransferEngine,
    checkpoint_files,
)


class DictSource(WeightSource):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    @cached_property
    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self.tensors.items()]

    def iter_full(self):
        yield from self.tensors.items()


class Target(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded: dict[str, torch.Tensor] = {}

    def load_weights(self, weights) -> set[str]:
        self.loaded.update(weights)
        return set(self.loaded)


class Client:
    """Forwards each call to the worker engine, as the HTTP routes would."""

    def __init__(self, worker: DiskWeightTransferEngine) -> None:
        self.worker = worker
        self.calls: list[str] = []
        self.files: list[str] = []

    def init_weight_transfer_engine(self, init_info) -> None:
        self.calls.append("init")

    def start_weight_update(self) -> None:
        self.calls.append("start")

    def update_weights(self, update_info) -> None:
        self.calls.append("update")
        self.files = sorted(os.listdir(update_info["path"]))
        self.worker.update_weights(update_info)

    def finish_weight_update(self, weight_version=None) -> None:
        self.calls.append("finish")


def make_worker(target: Target) -> DiskWeightTransferEngine:
    return DiskWeightTransferEngine(
        WeightTransferConfig(backend="disk"),
        VllmConfig(),
        torch.device("cpu"),
        target,
    )


def test_trainer_writes_shards_worker_loads_them(tmp_path):
    tensors = {
        "a": torch.randn(64, 64, dtype=torch.bfloat16),  # 8 KiB
        "b": torch.arange(10, dtype=torch.int64),
        "c": torch.randn(128, 64),  # 32 KiB: larger than a shard, written alone
        "d": torch.randn(3, 5, dtype=torch.float16).t(),  # non-contiguous
    }
    target = Target()
    client = Client(make_worker(target))
    trainer = DiskTrainerWeightTransferEngine.trainer_init(
        DiskTrainerInitInfo(
            rank=0, directory=str(tmp_path), shard_size_bytes=16 * 1024
        ),
        client=client,
        source=DictSource(tensors),
    )

    trainer.send_weights()

    assert client.calls == ["init", "start", "update", "finish"]
    assert len(client.files) == 3
    assert target.loaded.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert torch.equal(target.loaded[name], tensor)
        assert target.loaded[name].dtype == tensor.dtype
    assert os.listdir(tmp_path) == []


def test_index_selects_files(tmp_path):
    save_file({"w": torch.ones(2)}, tmp_path / "model-00001-of-00001.safetensors")
    save_file({"w": torch.zeros(2)}, tmp_path / "consolidated.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}})
    )
    assert [os.path.basename(f) for f in checkpoint_files(str(tmp_path))] == [
        "model-00001-of-00001.safetensors"
    ]
    target = Target()
    make_worker(target).update_weights({"path": str(tmp_path)})
    assert torch.equal(target.loaded["w"], torch.ones(2))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoint_files(str(tmp_path))


def test_truncated_checkpoint_is_rejected_before_loading(tmp_path):
    save_file({"a": torch.ones(64), "b": torch.ones(64)}, tmp_path / "x.safetensors")
    path = tmp_path / "x.safetensors"
    path.write_bytes(path.read_bytes()[:-16])
    target = Target()
    with pytest.raises(ValueError, match="truncated"):
        make_worker(target).update_weights({"path": str(tmp_path)})
    assert target.loaded == {}
