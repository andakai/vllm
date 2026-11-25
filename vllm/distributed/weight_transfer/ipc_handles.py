# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSON form of a CUDA IPC handle: the `rebuild_cuda_tensor` args that
`torch.multiprocessing.reductions.reduce_tensor` returns for a CUDA tensor.

HTTP carries this form, so the worker rebuilds the args from plain values and a
fixed set of torch classes instead of unpickling them.
"""

from typing import Any

import pybase64 as base64
import torch

# The positional args of `rebuild_cuda_tensor`, in order.
FIELDS = (
    "tensor_cls",
    "size",
    "stride",
    "tensor_offset",
    "storage_cls",
    "dtype",
    "device",
    "handle",
    "storage_size_bytes",
    "storage_offset_bytes",
    "requires_grad",
    "ref_counter_handle",
    "ref_counter_offset",
    "event_handle",
    "event_sync_required",
)
_BYTES = ("handle", "ref_counter_handle", "event_handle")
_INTS = (
    "tensor_offset",
    "device",
    "storage_size_bytes",
    "storage_offset_bytes",
    "ref_counter_offset",
)
_CLASSES = {
    f"{cls.__module__}.{cls.__qualname__}": cls
    for cls in (
        torch.Tensor,
        torch.nn.Parameter,
        torch.storage.TypedStorage,
        torch.storage.UntypedStorage,
    )
}


def _class_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def encode_ipc_args(args: tuple) -> dict[str, Any]:
    """JSON-native dict of one `reduce_tensor(tensor)[1]`."""
    out = dict(zip(FIELDS, args, strict=True))
    out["tensor_cls"] = _class_name(out["tensor_cls"])
    out["storage_cls"] = _class_name(out["storage_cls"])
    out["size"] = list(out["size"])
    out["stride"] = list(out["stride"])
    out["dtype"] = str(out["dtype"]).removeprefix("torch.")
    for key in _BYTES:
        if out[key] is not None:
            out[key] = base64.b64encode(out[key]).decode()
    return out


def decode_ipc_args(encoded: dict[str, Any]) -> tuple:
    """Inverse of `encode_ipc_args`; rejects anything else."""
    if encoded.keys() != set(FIELDS):
        raise ValueError(f"An IPC handle needs exactly the fields {FIELDS}")
    out = dict(encoded)
    for key in ("tensor_cls", "storage_cls"):
        if out[key] not in _CLASSES:
            raise ValueError(f"Unexpected {key} {out[key]!r} in an IPC handle")
        out[key] = _CLASSES[out[key]]
    dtype = getattr(torch, out["dtype"], None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unexpected dtype {out['dtype']!r} in an IPC handle")
    out["dtype"] = dtype
    out["size"] = torch.Size(int(n) for n in out["size"])
    out["stride"] = tuple(int(n) for n in out["stride"])
    for key in _INTS:
        out[key] = int(out[key])
    for key in ("requires_grad", "event_sync_required"):
        out[key] = bool(out[key])
    for key in _BYTES:
        if out[key] is not None:
            out[key] = base64.b64decode(out[key])
    return tuple(out[key] for key in FIELDS)


def is_encoded(value: Any) -> bool:
    return isinstance(value, dict) and "storage_cls" in value
