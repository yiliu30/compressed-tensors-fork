# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end demo of the compressed_tensors offload system.

Detects the available accelerator via torch.accelerator and exercises the major
offload features in order:
  1. Module-level offload (CPU <-> accelerator)
  2. Forward pass through an offloaded module
  3. disable_offloading / disable_onloading context managers
  4. update_offload_parameter
  5. Disk offload (safetensors-backed)
  6. Model-level offload (offload_model)
  7. Model-level dispatch (dispatch_model)
  8. Remove offload and restore module

All 8 tests work on any torch.accelerator-supported backend (CUDA, XPU, etc.).

Usage:
    /home/yiliu7/workspace/envs/ct/bin/python tasks/xpu-support-for-offload/demo.py
"""

import sys
import tempfile
import traceback

import torch
import torch.nn as nn

from compressed_tensors.offload import (
    disable_offloading,
    disable_onloading,
    get_execution_device,
    get_offloaded_device,
    update_offload_parameter,
)
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.dispatch import dispatch_model, offload_model
from compressed_tensors.offload.module import offload_module, remove_module_offload
from compressed_tensors.offload.utils import module_size


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_accelerator() -> torch.device:
    """Return the first available accelerator device, or exit."""
    if not torch.accelerator.is_available():
        print("ERROR: no accelerator found.")
        sys.exit(1)
    return torch.device(torch.accelerator.current_accelerator().type, 0)


def get_device_name(device: torch.device) -> str:
    try:
        return torch.accelerator.device_name(device.index or 0)
    except Exception:
        return str(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ONLOAD = get_accelerator()
OFFLOAD_CPU = torch.device("cpu")


def assert_device(tensor: torch.Tensor, expected: torch.device, label: str = ""):
    actual = tensor.device
    assert actual == expected, f"[{label}] expected {expected}, got {actual}"


def banner(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


# ---------------------------------------------------------------------------
# Simple model used for dispatch tests
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear0 = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear1(torch.relu(self.linear0(x)))


class TinyModel(nn.Module):
    _no_split_modules = ["Decoder"]

    def __init__(self, dim: int = 64):
        super().__init__()
        self.decoder0 = Decoder(dim)
        self.decoder1 = Decoder(dim)

    def forward(self, x):
        return self.decoder1(self.decoder0(x))


# ===========================================================================
#  1. Module-level CPU offload
# ===========================================================================


def test_module_offload():
    banner("1. Module-level CPU offload")

    linear = nn.Linear(128, 128, bias=True)
    # weights start on CPU
    assert_device(linear.weight, OFFLOAD_CPU, "before offload")

    offload_module(linear, onload_device=ONLOAD, offload_device=OFFLOAD_CPU)

    # after offloading, accessing .weight triggers onloading -> CUDA
    assert_device(linear.weight, ONLOAD, "after offload (onloaded)")
    assert isinstance(linear._parameters, OffloadCache)
    assert get_execution_device(linear) == ONLOAD
    assert get_offloaded_device(linear) == OFFLOAD_CPU

    print("  PASS: weights on CPU, onloaded to CUDA on access")


# ===========================================================================
#  2. Forward pass through an offloaded module
# ===========================================================================


def test_offloaded_forward():
    banner("2. Forward pass through offloaded module")

    linear = nn.Linear(128, 128, bias=True)
    offload_module(linear, onload_device=ONLOAD, offload_device=OFFLOAD_CPU)

    # input on CPU - the forward wrapper moves it to CUDA automatically
    x = torch.randn(4, 128, device=OFFLOAD_CPU)
    with torch.no_grad():
        y = linear(x)

    assert_device(y, ONLOAD, "output device")
    assert y.shape == (4, 128)

    print(f"  PASS: input on CPU -> output on {ONLOAD}, shape {tuple(y.shape)}")


# ===========================================================================
#  3. disable_offloading / disable_onloading
# ===========================================================================


def test_disable_offloading():
    banner("3a. disable_offloading (cache onloaded tensors)")

    linear = nn.Linear(32, 32)
    offload_module(linear, onload_device=ONLOAD, offload_device=OFFLOAD_CPU)

    with disable_offloading():
        w1 = linear.weight  # first access -> onloads
        w2 = linear.weight  # second access -> cache hit, same object
        assert w1 is w2, "expected cache hit inside disable_offloading"
        assert_device(w1, ONLOAD, "cached weight")

    print("  PASS: repeated access returns cached tensor")


def test_disable_onloading():
    banner("3b. disable_onloading (return raw offloaded tensors)")

    linear = nn.Linear(32, 32)
    offload_module(linear, onload_device=ONLOAD, offload_device=OFFLOAD_CPU)

    with disable_onloading():
        w = linear.weight
        assert_device(w, OFFLOAD_CPU, "raw offloaded weight")

    print("  PASS: disable_onloading returns CPU tensor directly")


# ===========================================================================
#  4. update_offload_parameter
# ===========================================================================


def test_update_parameter():
    banner("4. update_offload_parameter")

    linear = nn.Linear(32, 32)
    offload_module(linear, onload_device=ONLOAD, offload_device=OFFLOAD_CPU)

    new_data = torch.ones(32, 32)
    update_offload_parameter(linear, "weight", new_data)

    # verify the update took effect
    updated = linear.weight
    assert_device(updated, ONLOAD, "updated weight")
    assert torch.allclose(updated, new_data.to(ONLOAD)), "data mismatch after update"

    print("  PASS: parameter updated successfully via update_offload_parameter")


# ===========================================================================
#  5. Disk offload
# ===========================================================================


def test_disk_offload():
    banner("5. Disk offload (safetensors-backed)")

    linear = nn.Linear(64, 64, bias=True)
    original_weight = linear.weight.clone()

    with tempfile.TemporaryDirectory() as tmpdir:
        offload_module(
            linear,
            onload_device=ONLOAD,
            offload_device="disk",
            offload_dir=tmpdir,
        )

        # accessing triggers disk -> CUDA load
        w = linear.weight
        assert_device(w, ONLOAD, "disk-onloaded weight")
        assert torch.allclose(w, original_weight.to(ONLOAD)), "data mismatch"

        # forward pass works
        x = torch.randn(4, 64, device=OFFLOAD_CPU)
        with torch.no_grad():
            y = linear(x)
        assert_device(y, ONLOAD, "disk-offloaded forward output")

    print(f"  PASS: disk offload -> onload to {ONLOAD}, forward OK")


# ===========================================================================
#  6. offload_model (whole-model offload)
# ===========================================================================


def test_offload_model():
    banner("6. offload_model (whole model to CPU, execute on CUDA)")

    model = TinyModel()
    x = torch.randn(4, 64, device=OFFLOAD_CPU)

    # baseline: run on CPU
    with torch.no_grad():
        cpu_out = model(x)

    # offload everything to CPU, execute on CUDA
    offload_model(model, onload_device=ONLOAD)

    with torch.no_grad():
        offloaded_out = model(x)

    assert_device(offloaded_out, ONLOAD, "offloaded model output")

    # verify all leaf modules are offloaded
    for name, m in model.named_modules():
        if isinstance(m._parameters, OffloadCache):
            assert get_execution_device(m) == ONLOAD
            assert get_offloaded_device(m) == OFFLOAD_CPU

    # numerical check
    assert torch.allclose(offloaded_out, cpu_out.to(ONLOAD), atol=1e-5), (
        "numerical mismatch between CPU and offloaded execution"
    )

    print("  PASS: whole-model offload produces correct results")


# ===========================================================================
#  7. dispatch_model (automatic multi-device dispatch)
# ===========================================================================


def remove_module_offload_recursive(model: nn.Module):
    for m in model.modules():
        remove_module_offload(m, onload_tensors=False)


def test_dispatch_model():
    """
    dispatch_model places modules on-device as (device, device), which means
    OffloadCache.cls_from_device is called with the accelerator type. Now that
    production code uses torch.accelerator, this works on all backends.
    """
    banner("7. dispatch_model (auto dispatch with memory budget)")

    model = TinyModel()
    x = torch.randn(4, 64, device=OFFLOAD_CPU)

    with torch.no_grad():
        cpu_out = model(x)

    # give enough memory for the full model - everything stays on device
    total_size = module_size(model)
    device_memory = {ONLOAD: total_size * 2}

    dispatch_model(model, device_memory=device_memory, extra_memory=0)

    x_accel = x.to(ONLOAD)
    with torch.no_grad():
        dispatched_out = model(x_accel)

    assert_device(dispatched_out, ONLOAD, "dispatched output")
    assert torch.allclose(dispatched_out, cpu_out.to(ONLOAD), atol=1e-5)
    print("  PASS (full fit): all modules on device, correct output")

    # now halve the budget - some modules should be offloaded to CPU
    remove_module_offload_recursive(model)
    device_memory = {ONLOAD: total_size // 2}
    dispatch_model(model, device_memory=device_memory, extra_memory=0)

    with torch.no_grad():
        partial_out = model(x)

    assert_device(partial_out, ONLOAD, "partial-offload output")
    assert torch.allclose(partial_out, cpu_out.to(ONLOAD), atol=1e-5)

    # verify at least one module is offloaded to CPU
    has_offloaded = any(
        isinstance(m._parameters, OffloadCache)
        and get_offloaded_device(m) == OFFLOAD_CPU
        for m in model.modules()
    )
    assert has_offloaded, "expected at least one module offloaded to CPU"
    print("  PASS (partial fit): some modules offloaded to CPU, correct output")


# ===========================================================================
#  8. Remove offload and verify restoration
# ===========================================================================


def test_remove_offload():
    banner("8. Remove offload and restore module")

    linear = nn.Linear(64, 64)
    original_weight = linear.weight.clone()

    offload_module(linear, onload_device=ONLOAD, offload_device=OFFLOAD_CPU)
    assert isinstance(linear._parameters, OffloadCache)

    remove_module_offload(linear, onload_tensors=True)
    assert not isinstance(linear._parameters, OffloadCache)
    assert_device(linear.weight, ONLOAD, "restored weight on CUDA")
    assert torch.allclose(linear.weight, original_weight.to(ONLOAD))

    print("  PASS: offload removed, weights restored to CUDA")


# ===========================================================================
#  Main
# ===========================================================================


def main():
    print(f"Using device: {ONLOAD}  ({get_device_name(ONLOAD)})")

    tests = [
        test_module_offload,
        test_offloaded_forward,
        test_disable_offloading,
        test_disable_onloading,
        test_update_parameter,
        test_disk_offload,
        test_offload_model,
        test_dispatch_model,
        test_remove_offload,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception:
            failed += 1
            traceback.print_exc()

    banner(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")


if __name__ == "__main__":
    main()
