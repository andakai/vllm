# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The inline engine round trip: the trainer chunks and encodes the source's
tensors, and the worker engine loads every chunk into its target."""

from functools import cached_property

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.inline_engine import (
    InlineTrainerInitInfo,
    InlineTrainerWeightTransferEngine,
    InlineWeightTransferEngine,
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

    def __init__(self, worker: InlineWeightTransferEngine) -> None:
        self.worker = worker
        self.calls: list[str] = []

    def init_weight_transfer_engine(self, init_info) -> None:
        self.calls.append("init")

    def start_weight_update(self) -> None:
        self.calls.append("start")

    def update_weights(self, update_info) -> None:
        self.calls.append("update")
        self.worker.update_weights(update_info)

    def finish_weight_update(self, weight_version=None) -> None:
        self.calls.append("finish")


def test_tensors_arrive_unchanged_in_bounded_chunks():
    tensors = {
        "a": torch.randn(64, 64, dtype=torch.bfloat16),  # 8 KiB
        "b": torch.arange(10, dtype=torch.int64),
        "c": torch.randn(128, 64),  # 32 KiB: larger than a chunk, sent alone
        "d": torch.randn(3, 5, dtype=torch.float16).t(),  # non-contiguous
    }
    target = Target()
    worker = InlineWeightTransferEngine(
        WeightTransferConfig(backend="inline"),
        VllmConfig(),
        torch.device("cpu"),
        target,
    )
    client = Client(worker)
    trainer = InlineTrainerWeightTransferEngine.trainer_init(
        InlineTrainerInitInfo(rank=0, chunk_size_bytes=16 * 1024),
        client=client,
        source=DictSource(tensors),
    )

    trainer.send_weights()

    assert client.calls == ["init", "start", "update", "update", "update", "finish"]
    assert target.loaded.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert torch.equal(target.loaded[name], tensor)
        assert target.loaded[name].dtype == tensor.dtype
