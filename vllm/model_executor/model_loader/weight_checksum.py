# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-parameter weight digests and reset, for verifying weight updates."""

import hashlib
from collections.abc import Collection, Iterator

import torch
import torch.nn as nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from vllm.model_executor.parameter import SharedWeightParameter


def _tensors(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    # Both keep their bytes in inner tensors (e.g. TorchAO's qdata and scale).
    if isinstance(tensor, SharedWeightParameter):
        for index, partition in tensor.partitions.items():
            yield from _tensors(f"{name}.{index}", partition)
    elif is_traceable_wrapper_subclass(tensor):
        for attr in tensor.__tensor_flatten__()[0]:
            yield from _tensors(f"{name}.{attr}", getattr(tensor, attr))
    else:
        yield name, tensor


def _weights(
    model: nn.Module, names: Collection[str] | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        if names is None or name in names:
            yield from _tensors(name, param)


def compute_tensor_digests(
    model: nn.Module, names: Collection[str] | None = None
) -> dict[str, str]:
    """Digest every parameter, or only the parameters named in `names`."""
    return {
        name: hashlib.sha256(
            weight.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, weight in _weights(model, None if names is None else set(names))
    }


def zero_weights(model: nn.Module) -> None:
    for _, weight in _weights(model):
        weight.data.zero_()
