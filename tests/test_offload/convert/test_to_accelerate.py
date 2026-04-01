# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
from compressed_tensors.offload import dispatch_with_map, offload_module, to_accelerate
from compressed_tensors.offload.convert.to_accelerate import to_accelerate_module
from tests.test_offload.conftest import torchrun
from tests.testing_utils import requires_gpu


acclerate = pytest.importorskip("accelerate")

_ACCEL_TYPE = torch.accelerator.current_accelerator().type
_DEV0 = f"{_ACCEL_TYPE}:0"


@pytest.mark.unit
@requires_gpu
@pytest.mark.parametrize(
    "offload_device", [_ACCEL_TYPE, _DEV0, "cpu", "disk"]
)
def test_to_accelerate_module(offload_device):
    linear = torch.nn.Linear(5, 5)
    offload_module(linear, _ACCEL_TYPE, offload_device)

    _offload_device = to_accelerate_module(linear, name="", hf_disk_index={})
    if offload_device == _ACCEL_TYPE:
        assert _offload_device == _DEV0
    else:
        assert _offload_device == offload_device


@pytest.mark.unit
@requires_gpu
def test_to_accelerate(accel_device, tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    _accel_dev = torch.device(_ACCEL_TYPE)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 5), torch.nn.Linear(5, 5), torch.nn.Linear(5, 5)
    )
    device_map = {
        "0": (_accel_dev, torch.device("cpu")),
        "1": (_accel_dev, _accel_dev),
        "2": (_accel_dev, "disk"),
    }
    dispatch_with_map(model, device_map, offload_dir)

    hf_device_map = to_accelerate(model)
    assert hf_device_map == {"": "cpu", "0": "cpu", "1": str(accel_device), "2": "disk"}
    assert hasattr(model[0], "_hf_hook")
    assert hasattr(model[1], "_hf_hook")
    assert hasattr(model[2], "_hf_hook")


@pytest.mark.unit
@requires_gpu(2)
@torchrun(world_size=2)
def test_to_accelerate_dist(accel_device, tmp_path):
    test_to_accelerate(accel_device, tmp_path)
