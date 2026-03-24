# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Quick standalone smoke test for the torch.accelerator offload migration.

Exercises the core offload paths on whatever accelerator is available:
  1. CPU offload + forward pass
  2. Disk offload + forward pass
  3. dispatch_model (full fit + partial fit with CPU spillover)

Usage:
    /home/yiliu7/workspace/envs/ct/bin/python tasks/xpu-support-for-offload/quick_demo.py
"""

import logging
import sys
import tempfile

import torch
import torch.nn as nn
from loguru import logger

from compressed_tensors.offload import disable_offloading, get_offloaded_device
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.dispatch import dispatch_model
from compressed_tensors.offload.module import offload_module, remove_module_offload
from compressed_tensors.offload.utils import module_size

# Suppress loguru warnings from dispatch_model so test output stays clean.
# The "Forced to offload" warning is expected in test_dispatch_partial since we
# intentionally halve the memory budget to exercise the CPU-spillover path.
logger.disable("compressed_tensors")


# ── device detection ─────────────────────────────────────────────────────

if not torch.accelerator.is_available():
    print("ERROR: no accelerator found")
    sys.exit(1)

DEVICE = torch.device(torch.accelerator.current_accelerator().type, 0)
CPU = torch.device("cpu")

try:
    dev_name = getattr(torch, DEVICE.type).get_device_name(0)
except Exception:
    dev_name = str(DEVICE)
print(f"Accelerator: {DEVICE}  ({dev_name})")


# ── tiny model ───────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return torch.relu(self.fc(x))


class TinyModel(nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self, d=64):
        super().__init__()
        self.b0 = Block(d)
        self.b1 = Block(d)

    def forward(self, x):
        return self.b1(self.b0(x))


# ── helpers ──────────────────────────────────────────────────────────────

passed = failed = 0


def run(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as exc:
        failed += 1
        print(f"  FAIL  {name}: {exc}")


# ── tests ────────────────────────────────────────────────────────────────

def test_cpu_offload():
    m = TinyModel()
    x = torch.randn(2, 64)
    with torch.no_grad():
        ref = m(x)

    for mod in m.modules():
        if list(mod.parameters(recurse=False)):
            offload_module(mod, onload_device=DEVICE, offload_device=CPU)

    with torch.no_grad():
        out = m(x)

    assert out.device == DEVICE
    assert torch.allclose(out, ref.to(DEVICE), atol=1e-5)


def test_disk_offload():
    linear = nn.Linear(64, 64)
    w = linear.weight.clone()

    with tempfile.TemporaryDirectory() as d:
        offload_module(linear, onload_device=DEVICE, offload_device="disk",
                       offload_dir=d)
        assert linear.weight.device == DEVICE
        assert torch.allclose(linear.weight, w.to(DEVICE))

        with torch.no_grad():
            y = linear(torch.randn(2, 64))
        assert y.device == DEVICE


def test_dispatch_full():
    m = TinyModel()
    x = torch.randn(2, 64)
    with torch.no_grad():
        ref = m(x)

    mem = {DEVICE: module_size(m) * 4}
    dispatch_model(m, device_memory=mem, extra_memory=0)

    with torch.no_grad():
        out = m(x.to(DEVICE))
    assert out.device == DEVICE
    assert torch.allclose(out, ref.to(DEVICE), atol=1e-5)


def test_dispatch_partial():
    m = TinyModel()
    x = torch.randn(2, 64)
    with torch.no_grad():
        ref = m(x)

    mem = {DEVICE: module_size(m) // 2}
    dispatch_model(m, device_memory=mem, extra_memory=0)

    with torch.no_grad():
        out = m(x)
    assert out.device == DEVICE
    assert torch.allclose(out, ref.to(DEVICE), atol=1e-5)

    assert any(
        isinstance(mod._parameters, OffloadCache)
        and get_offloaded_device(mod) == CPU
        for mod in m.modules()
    ), "expected at least one CPU-offloaded module"


# ── main ─────────────────────────────────────────────────────────────────

run("cpu_offload + forward", test_cpu_offload)
run("disk_offload + forward", test_disk_offload)
run("dispatch_model (full)", test_dispatch_full)
run("dispatch_model (partial)", test_dispatch_partial)

print(f"\n{passed}/{passed + failed} passed")
sys.exit(1 if failed else 0)
