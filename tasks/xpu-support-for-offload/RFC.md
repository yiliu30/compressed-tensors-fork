# RFC: Multi-Accelerator Offload Support via `torch.accelerator`

## Summary

Replace hardcoded `torch.cuda.*` calls in the offload stack with PyTorch's
`torch.accelerator` API. This unlocks XPU (and NPU) support with minimal
changes — no new abstraction layers, no new files.

## Motivation

The `compressed_tensors` offload system is designed to move model parameters
between CPU, disk, and accelerator memory during quantization and inference.
Most of the system is already device-agnostic — cache implementations, tensor
movement, and the dispatch pipeline all work through `tensor.to(device)` and
don't care what the target device is.

However, the device detection and memory query layer is hardcoded to CUDA.
On `main` today there are **9 `torch.cuda.*` calls** across 5 files in
`src/compressed_tensors/offload/`:

- `dispatch.py` and `load.py` use `torch.cuda.is_available()`,
  `torch.cuda.device_count()`, and `torch.cuda.get_device_properties()` to
  discover accelerators and query their memory.
- `cache/base.py` and `cache/disk.py` match on the literal string `"cuda"` to
  route tensors to the right cache class and resolve device indices.
- `dist_utils.py` hardcodes `backend="nccl"` and `torch.cuda.set_device()`.
- `convert/helpers.py` checks `device.type == "cuda"` for device normalization.

This means any non-CUDA accelerator — Intel XPU, Ascend NPU, or others — simply
cannot use the offload system. The `dispatch_model()` call falls back to
CPU-only because `torch.cuda.is_available()` returns `False`, even when
XPU devices are present and working.

With growing adoption of Intel XPU (Arc, Data Center GPU Max, Gaudi) and Ascend
NPU for model quantization and inference, this is a real blocker. The good news
is that the fix is small: **~35 lines across 6 production files** turns
CUDA-only offload into multi-accelerator offload.

## Approach: `torch.accelerator`

PyTorch 2.6 introduced [`torch.accelerator`](https://pytorch.org/docs/stable/generated/torch.accelerator.html),
a built-in abstraction over hardware backends. Each `torch.cuda.*` call we use
has a direct `torch.accelerator.*` equivalent — for example,
`torch.cuda.device_count()` becomes `torch.accelerator.device_count()`. The API
auto-detects the active backend (CUDA, XPU, etc.) at runtime, so the same code
works everywhere.

Backends supported out of the box:

- **CUDA** — built-in
- **XPU** — built-in since PyTorch 2.4
- **Ascend NPU** — registers via `torch.utils.rename_privateuse1_backend("npu")`

For distributed, PyTorch maintains a backend mapping (CUDA→NCCL, XPU→XCCL,
NPU→HCCL), accessible via `dist.get_default_backend_for_device()`.

## What Changes

All changes are in the offload module (`src/compressed_tensors/offload/`).

**Device detection and memory queries** (`dispatch.py`, `load.py`) — Replace
`torch.cuda.is_available()`, `torch.cuda.device_count()`, and
`torch.cuda.get_device_properties(i).total_memory` with their
`torch.accelerator` equivalents.

**Cache device routing** (`cache/base.py`, `cache/disk.py`) — Replace hardcoded
`"cuda"` match arms with a capability-based `is_accelerator_type()` helper that
validates the device type against `torch.accelerator.current_accelerator()`.
This avoids both a brittle allowlist and an overly permissive exclusion list.

**Distributed init** (`dist_utils.py`) — Replace `backend="nccl"` and
`torch.cuda.set_device()` with `dist.get_default_backend_for_device()` and
`torch.accelerator.set_device_index()`.

**Device normalization** (`convert/helpers.py`) — Generalize the
`device.type == "cuda"` check to use `is_accelerator_type()`.

**Safetensors compatibility** (`cache/disk.py`) — Add a scoped CPU fallback in
`DiskCache.onload()` for backends whose device strings aren't yet recognized by
safetensors. The fallback catches only device-resolution errors
(`SafetensorError`, `AssertionError`) during `safe_open()` and retries on CPU.

**Tests** (~14 files) — Migrate CUDA-specific fixtures (`cuda_device` →
`accel_device`), update the `requires_gpu` decorator to use
`torch.accelerator.is_available()`, and repoint mock targets from `torch.cuda`
to `torch.accelerator`.

## Support Policy

- **Minimum PyTorch version:** Bump to `torch>=2.6.0` in `setup.py`. No
  compatibility shim — all downstream consumers (vLLM, llm-compressor) already
  require ≥2.1, and any XPU/NPU user is on ≥2.6.
- **CUDA:** Tier 1 — fully supported, validated by existing CI.
- **XPU:** Tier 1 — fully supported, validated on PyTorch 2.10.0+xpu with
  4× Intel Arc Pro B60.
- **NPU:** Tier 2 — expected to work, not yet CI-validated. Bug reports welcome.

## Implementation Steps

1. **Add `is_accelerator_type()` helper** — A small utility that checks whether
   a device type string matches the currently active PyTorch accelerator backend.
   Lives in `offload/cache/base.py` (or a shared utils module).

2. **Update production code** (6 files, ~35 lines) — Replace `torch.cuda.*`
   calls and `"cuda"` literals in `cache/base.py`, `cache/disk.py`,
   `dispatch.py`, `load.py`, `dist_utils.py`, and `convert/helpers.py`.

3. **Bump torch minimum** — Update `setup.py` from `torch>=1.7.0` to
   `torch>=2.6.0`.

4. **Migrate test harness** (~14 files, ~40 lines) — Introduce `accel_device`
   fixture, update `requires_gpu` decorator, replace mock targets in
   `test_dispatch.py`, and rename `cuda_device` across offload test files.

5. **Validate** — Run the full offload test suite on both CUDA and XPU,
   including single-device offload, disk offload, `dispatch_model` multi-device,
   safetensors direct-load, distributed init + broadcast, and a full W4A16
   quantization pipeline.

## Risks

- **PyTorch version bump** — Raising the minimum to 2.6.0 could affect users on
  older PyTorch. Mitigated by the fact that the offload stack itself is recent
  and all major consumers already require ≥2.1.
- **Safetensors device strings** — Not all backends are verified. Mitigated by
  the scoped CPU fallback in `DiskCache.onload()`.
- **Distributed backend maturity** — XCCL (XPU) and HCCL (NPU) vary in
  maturity. Errors are not masked — they propagate clearly from PyTorch.
