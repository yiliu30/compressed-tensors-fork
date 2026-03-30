# RFC: Multi-Accelerator Offload Support via `torch.accelerator`

## Summary

Replace hardcoded `torch.cuda.*` calls in the offload stack with PyTorch's
`torch.accelerator` API. This unlocks XPU (and NPU) support with minimal
changes — no new abstraction layers, no new files.

---

## Motivation

The `compressed_tensors` offload system moves model parameters between CPU,
disk, and accelerator memory during quantization and inference. Most of the
system is already device-agnostic — cache implementations, tensor movement,
and the dispatch pipeline all work through `tensor.to(device)` and don't care
what the target device is.

The problem is in the **device detection and memory query layer**, which is
hardcoded to CUDA. On `main` today:

| Area | Files | CUDA-specific code |
|------|-------|--------------------|
| Device discovery & memory | `dispatch.py`, `load.py` | `torch.cuda.is_available()`, `device_count()`, `get_device_properties()` |
| Cache routing | `cache/base.py`, `cache/disk.py` | Literal `"cuda"` string matching |
| Distributed init | `dist_utils.py` | `backend="nccl"`, `torch.cuda.set_device()` |
| Device normalization | `convert/helpers.py` | `device.type == "cuda"` |

> **9 `torch.cuda.*` calls** across 5 files + hardcoded `"cuda"` strings in a 6th.

This means any non-CUDA accelerator — Intel XPU, Ascend NPU, or others —
**cannot use the offload system at all**. `dispatch_model()` falls back to
CPU-only because `torch.cuda.is_available()` returns `False`, even when XPU
devices are present and working.

With growing adoption of Intel XPU (Arc, Data Center GPU Max, Gaudi) and
Ascend NPU for model quantization and inference, this is a real blocker.
The good news: **~35 lines across 6 files** turns CUDA-only offload into
multi-accelerator offload.

---

## Approach: `torch.accelerator`

PyTorch 2.6 introduced
[`torch.accelerator`](https://pytorch.org/docs/stable/generated/torch.accelerator.html) —
a built-in abstraction over hardware backends. Each `torch.cuda.*` call we use
has a direct `torch.accelerator.*` equivalent (e.g., `torch.cuda.device_count()`
→ `torch.accelerator.device_count()`). The API auto-detects the active backend
at runtime, so the same code works everywhere.

**Supported backends:**

| Backend | Registration |
|---------|-------------|
| CUDA | Built-in |
| XPU | Built-in (since PyTorch 2.4) |
| Ascend NPU | Via `torch.utils.rename_privateuse1_backend("npu")` |

For distributed, PyTorch maintains a backend mapping (CUDA → NCCL, XPU → XCCL,
NPU → HCCL) accessible via `dist.get_default_backend_for_device()`.

---

## What Changes

All changes are in the offload module (`src/compressed_tensors/offload/`):

| Change | Files | Description |
|--------|-------|-------------|
| Device detection & memory | `dispatch.py`, `load.py` | `torch.cuda.*` → `torch.accelerator.*` for availability, device count, and memory queries |
| Cache routing | `cache/base.py`, `cache/disk.py` | `"cuda"` match arms → capability-based `is_accelerator_type()` helper |
| Distributed init | `dist_utils.py` | `"nccl"` → `dist.get_default_backend_for_device()` |
| Device normalization | `convert/helpers.py` | `device.type == "cuda"` → `is_accelerator_type()` |
| Safetensors compat | `cache/disk.py` | Scoped CPU fallback in `DiskCache.onload()` for unrecognized device strings |
| Tests | ~14 files | `cuda_device` → `accel_device` fixture, `requires_gpu` update, mock retargeting |

---

## Support Policy

| Backend | Tier | Status |
|---------|------|--------|
| CUDA | 1 | Fully supported, validated by existing CI |
| XPU | 1 | Fully supported, validated on PyTorch 2.10.0+xpu (4× Intel Arc Pro B60) |
| NPU | 2 | Expected to work, not yet CI-validated — bug reports welcome |

- **Minimum PyTorch version:** `torch>=2.6.0` (no compatibility shim). All
  downstream consumers (vLLM, llm-compressor) already require ≥2.1, and any
  XPU/NPU user is on ≥2.6.

---

## Implementation Steps

| Step | What | Scope |
|------|------|-------|
| 1 | Add `is_accelerator_type()` helper | Small utility to validate device types against active accelerator |
| 2 | Update production code | 6 files, ~35 lines — replace `torch.cuda.*` and `"cuda"` literals |
| 3 | Bump torch minimum | `setup.py`: `torch>=1.7.0` → `torch>=2.6.0` |
| 4 | Migrate test harness | ~14 files, ~40 lines — new fixture, decorator, mock targets |
| 5 | Validate | Full test suite on CUDA + XPU (see matrix below) |

**Validation matrix:**

| Scenario | CUDA | XPU |
|----------|:----:|:---:|
| Single-device offload (CPU ↔ accel) | ✓ | ✓ |
| Disk offload (disk ↔ accel) | ✓ | ✓ |
| `dispatch_model` multi-device | ✓ | ✓ |
| `safetensors` direct-load | ✓ | ✓ |
| Distributed init + broadcast | ✓ | ✓ |
| Full W4A16 quantization pipeline | ✓ | ✓ |

---

## Risks

| Risk | Mitigation |
|------|------------|
| **PyTorch version bump** — 2.6.0 minimum could affect users on older versions | Offload stack is recent; all major consumers already require ≥2.1 |
| **Safetensors device strings** — not all backends verified | Scoped CPU fallback in `DiskCache.onload()` catches only device errors |
| **Distributed maturity** — XCCL/HCCL vary in maturity | Errors propagate clearly from PyTorch, not masked |
