# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight transfer that carries the tensors in the update requests.

The trainer serializes each chunk with safetensors and sends it as base64 in
`update_weights`, so the transfer needs no data-plane rendezvous, no shared
filesystem and no pickling: only the control plane (HTTP or Ray) between the
trainer and the server.
"""

from dataclasses import dataclass
from typing import ClassVar, Self

import pybase64 as base64
import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

# Base64 makes a request about 4/3 of this, within the Rust frontend's default
# 32 MiB `VLLM_HTTP_MAX_REQUEST_BODY_SIZE`.
DEFAULT_CHUNK_SIZE_BYTES = 16 * 1024 * 1024


@dataclass
class InlineWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for inline weight transfer: nothing to set up."""


@dataclass
class InlineTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for inline weight transfer."""

    backend: ClassVar[str] = "inline"

    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES
    """Tensor bytes per `update_weights` request, before base64 encoding, which
    adds a third. A tensor larger than this is sent alone."""


@dataclass
class InlineWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """One chunk of an inline weight transfer."""

    safetensors_b64: str
    """The chunk's tensors in safetensors format, base64-encoded."""


def encode_tensors(tensors: dict[str, torch.Tensor]) -> str:
    return base64.b64encode(save_safetensors(tensors)).decode("ascii")


class InlineWeightTransferEngine(
    WeightTransferEngine[InlineWeightTransferInitInfo, InlineWeightTransferUpdateInfo]
):
    """Loads the tensors carried by each update request into the model."""

    init_info_cls = InlineWeightTransferInitInfo
    update_info_cls = InlineWeightTransferUpdateInfo

    def init_transfer_engine(self, init_info: InlineWeightTransferInitInfo) -> None:
        pass

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)

    def receive_weights(self, update_info: InlineWeightTransferUpdateInfo) -> None:
        tensors = load_safetensors(base64.b64decode(update_info.safetensors_b64))
        self.model.load_weights(tensors.items())

    def shutdown(self) -> None:
        pass


class InlineTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[InlineTrainerInitInfo]
):
    """Sends the source's tensors through the client, in chunks of about
    `chunk_size_bytes`.

    Every rank iterates the source, which may run collectives; only the sender
    copies the tensors to the CPU and sends them, so it must own every
    parameter.
    """

    init_info_cls = InlineTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.chunk_size_bytes = chunk_size_bytes

    @classmethod
    def trainer_init(
        cls,
        init_info: InlineTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("Inline weight transfer requires a WeightSource.")
        if not source.implements_iter_full():
            raise TypeError(
                f"Inline transfer sends full tensors, but {type(source).__name__} "
                "does not implement iter_full()."
            )
        if init_info.is_sender and len(source.owned_names(init_info.rank)) != len(
            source.metadata
        ):
            raise ValueError(
                "The sending rank must own every parameter for inline transfer."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            chunk_size_bytes=init_info.chunk_size_bytes,
        )
        if engine.is_sender:
            engine.client.init_weight_transfer_engine({})
        return engine

    def send_weights(self) -> None:
        assert self.source is not None
        if not self.is_sender:
            for _ in self.source.iter_full():
                pass
            return
        self.client.start_weight_update()
        chunk: dict[str, torch.Tensor] = {}
        chunk_bytes = 0
        for name, tensor in self.source.iter_full():
            cpu_tensor = tensor.detach().to("cpu").contiguous()
            if chunk and chunk_bytes + cpu_tensor.nbytes > self.chunk_size_bytes:
                self._send_chunk(chunk)
                chunk, chunk_bytes = {}, 0
            chunk[name] = cpu_tensor
            chunk_bytes += cpu_tensor.nbytes
        if chunk:
            self._send_chunk(chunk)
        self.client.finish_weight_update()

    def _send_chunk(self, chunk: dict[str, torch.Tensor]) -> None:
        self.client.update_weights({"safetensors_b64": encode_tensors(chunk)})
