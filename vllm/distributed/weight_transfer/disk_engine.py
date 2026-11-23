# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight transfer through safetensors checkpoints on shared storage.

Each `update_weights` names a checkpoint directory (or file) that every worker
can read; the workers load it with the same safetensors iterator as a cold
start. The trainer either writes the checkpoint itself (for example with
`save_pretrained`) and calls the routes, or uses the trainer engine here,
which writes the source's tensors and removes them once loaded.
"""

import glob
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import ClassVar, Self

import torch
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

DEFAULT_SHARD_SIZE_BYTES = 4 * 1024**3


@dataclass
class DiskWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for disk weight transfer: nothing to set up."""


@dataclass
class DiskTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for disk weight transfer."""

    backend: ClassVar[str] = "disk"

    directory: str = ""
    """Directory on storage shared with every inference worker; each update
    writes a fresh subdirectory here and removes it once loaded."""

    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES
    """Tensor bytes per safetensors file. A larger tensor gets its own file."""


@dataclass
class DiskWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """A checkpoint to load."""

    path: str
    """A directory of `*.safetensors` files, or one `.safetensors` file. In a
    directory with a `model.safetensors.index.json`, only the indexed files
    are loaded."""


def checkpoint_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {path!r}.")
    if os.path.isfile(os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)):
        from vllm.model_executor.model_loader.weight_utils import (
            filter_duplicate_safetensors_files,
        )

        files = filter_duplicate_safetensors_files(files, path, SAFE_WEIGHTS_INDEX_NAME)
    return files


class DiskWeightTransferEngine(
    WeightTransferEngine[DiskWeightTransferInitInfo, DiskWeightTransferUpdateInfo]
):
    """Loads the checkpoint each update request names into the model."""

    init_info_cls = DiskWeightTransferInitInfo
    update_info_cls = DiskWeightTransferUpdateInfo

    def init_transfer_engine(self, init_info: DiskWeightTransferInitInfo) -> None:
        pass

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)

    def receive_weights(self, update_info: DiskWeightTransferUpdateInfo) -> None:
        from vllm.model_executor.model_loader.weight_utils import (
            safetensors_weights_iterator,
        )

        files = checkpoint_files(update_info.path)
        self.model.load_weights(
            safetensors_weights_iterator(files, use_tqdm_on_load=False)
        )

    def shutdown(self) -> None:
        pass


class DiskTrainerWeightTransferEngine(TrainerWeightTransferEngine[DiskTrainerInitInfo]):
    """Writes the source's tensors as safetensors shards under `directory`,
    has the server load them, and removes them.

    Every rank iterates the source, which may run collectives; only the sender
    writes, so it must own every parameter.
    """

    init_info_cls = DiskTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        directory: str,
        is_sender: bool = True,
        shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.directory = directory
        self.shard_size_bytes = shard_size_bytes

    @classmethod
    def trainer_init(
        cls,
        init_info: DiskTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("Disk weight transfer requires a WeightSource.")
        if not init_info.directory:
            raise ValueError("Disk weight transfer requires a shared directory.")
        if not source.implements_iter_full():
            raise TypeError(
                f"Disk transfer writes full tensors, but {type(source).__name__} "
                "does not implement iter_full()."
            )
        if init_info.is_sender and len(source.owned_names(init_info.rank)) != len(
            source.metadata
        ):
            raise ValueError(
                "The sending rank must own every parameter for disk transfer."
            )
        engine = cls(
            client=client,
            source=source,
            directory=init_info.directory,
            is_sender=init_info.is_sender,
            shard_size_bytes=init_info.shard_size_bytes,
        )
        if engine.is_sender:
            os.makedirs(engine.directory, exist_ok=True)
            engine.client.init_weight_transfer_engine({})
        return engine

    def send_weights(self) -> None:
        assert self.source is not None
        if not self.is_sender:
            for _ in self.source.iter_full():
                pass
            return
        path = tempfile.mkdtemp(prefix="update-", dir=self.directory)
        try:
            shard: dict[str, torch.Tensor] = {}
            shard_bytes = num_shards = 0
            for name, tensor in self.source.iter_full():
                cpu_tensor = tensor.detach().to("cpu").contiguous()
                if shard and shard_bytes + cpu_tensor.nbytes > self.shard_size_bytes:
                    num_shards = self._write(path, shard, num_shards)
                    shard, shard_bytes = {}, 0
                shard[name] = cpu_tensor
                shard_bytes += cpu_tensor.nbytes
            if shard:
                self._write(path, shard, num_shards)
            self.client.start_weight_update()
            self.client.update_weights({"path": path})
            self.client.finish_weight_update()
        finally:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _write(path: str, shard: dict[str, torch.Tensor], index: int) -> int:
        save_file(shard, os.path.join(path, f"shard-{index:05d}.safetensors"))
        return index + 1
