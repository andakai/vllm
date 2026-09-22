# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.glm5next.common import kda as glm_kda
from vllm.models.kimi_k3.nvidia import kda as kimi_kda

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize(
    ("kda_module", "layer_cls", "num_inputs"),
    [
        (glm_kda, glm_kda.Glm5NextLinearAttention, 4),
        (kimi_kda, kimi_kda.KimiK3DeltaAttention, 5),
    ],
    ids=["glm5next", "kimi_k3"],
)
@pytest.mark.parametrize(
    "attn_metadata", [None, {}], ids=["no_metadata", "missing_layer_metadata"]
)
@pytest.mark.parametrize("flashkda", [True, False], ids=["flashkda", "other_backend"])
def test_missing_metadata_reserves_flashkda_workspace(
    monkeypatch,
    kda_module,
    layer_cls,
    num_inputs,
    attn_metadata,
    flashkda,
):
    specs = (
        ((2, 3, 4), torch.bfloat16),
        ((128,), torch.uint8),
    )
    workspace_manager = Mock()
    get_workspace_manager = Mock(return_value=workspace_manager)
    monkeypatch.setattr(
        kda_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=attn_metadata),
    )
    monkeypatch.setattr(kda_module, "current_workspace_manager", get_workspace_manager)

    layer = object.__new__(layer_cls)
    object.__setattr__(layer, "prefix", "model.layers.0.self_attn")
    object.__setattr__(layer, "_flashkda_buffer_specs", specs if flashkda else None)
    empty = torch.empty(0)

    assert layer._forward(*([empty] * num_inputs)) is None
    if flashkda:
        get_workspace_manager.assert_called_once_with()
        workspace_manager.reserve_simultaneous.assert_called_once_with(*specs)
    else:
        get_workspace_manager.assert_not_called()
