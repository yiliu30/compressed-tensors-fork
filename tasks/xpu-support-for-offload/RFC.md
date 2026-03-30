# RFC: Multi-Accelerator Offload Support via `torch.accelerator`

## Summary

Replace all hardcoded `torch.cuda.*` calls in `compressed_tensors.offload` with
`torch.accelerator` (PyTorch 2.6+). This enables XPU, Ascend NPU, Gaudi HPU, and
any future backend — with **zero new abstraction code** and no new files.

## Support Policy

### Minimum PyTorch Version

`setup.py` currently declares `torch>=1.7.0`. The `torch.accelerator` module was
introduced in PyTorch 2.6. This RFC proposes:

- **Bump the effective minimum to `torch>=2.6.0` for offload functionality.**
  Any user targeting XPU, NPU, or HPU already requires ≥2.6. CUDA users on
  older PyTorch versions do not use the offload stack in practice (it was
  added recently and depends on features unavailable before 2.x).
- **No compatibility shim.** A fallback path that re-imports `torch.cuda` when
  `torch.accelerator` is missing would double the maintenance surface and is
  not justified — every downstream consumer (vLLM, llm-compressor) already
  requires PyTorch ≥2.1.
- **Release impact:** Update `setup.py` to declare `torch>=2.6.0` so the
  constraint is explicit rather than a silent runtime failure.

### Supported Backends at Merge

| Backend | Status | Validated |
|---------|--------|-----------|
| CUDA    | Tier 1 — fully supported, existing CI | Yes (existing tests) |
| XPU     | Tier 1 — fully supported | Yes (PyTorch 2.10.0+xpu, 4× Intel Arc Pro B60) |
| Ascend NPU | Tier 2 — expected to work, not CI-validated | No |
| Gaudi HPU  | Tier 2 — expected to work, not CI-validated | No |
| Apple MPS  | Tier 2 — expected to work, not CI-validated | No |

Tier 2 backends are not blocked from working; they simply lack CI validation at
merge time. Bug reports from Tier 2 users are welcome and will be treated as
regular issues.

## Why `torch.accelerator`

All `torch.accelerator` functions delegate to C++ registered backends. Any device
that registers with PyTorch (via native support or `PrivateUse1` rename) gets
full support automatically:

- **CUDA** — built-in
- **XPU** — built-in since PyTorch 2.4
- **Ascend NPU** — `import torch_npu` registers via `rename_privateuse1_backend("npu")`
- **Gaudi HPU** — `import habana_frameworks.torch` registers `"hpu"`
- **Apple MPS** — built-in

### API Mapping (verified on PyTorch 2.10.0+xpu, 4× Intel Arc Pro B60)

| Current code | Replacement |
|---|---|
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` |
| `torch.cuda.set_device(dev)` | `torch.accelerator.set_device_index(idx)` |
| `torch.cuda.get_device_properties(i).total_memory` | `torch.accelerator.get_memory_info(i)[1]` |
| `torch.device(f"cuda:{idx}")` | `torch.device(accel_type, idx)` |
| `backend="nccl"` | `dist.get_default_backend_for_device(torch.device(accel_type))` |

The distributed backend is resolved via the public API
`dist.get_default_backend_for_device()`, which queries PyTorch's registered
backend model: `cuda→nccl`, `xpu→xccl`, `npu→hccl`, `mps→gloo`.

## Scope

The offload system is 90% device-agnostic already. All cache implementations
(`CPUCache`, `DeviceCache`, `DiskCache`, and distributed variants) work for any
device via `tensor.to()`. Changes are confined to **6 production files, ~35 lines**.

## Device Classification: Capability-Based Approach

The original draft used an exclusion-based helper (`device_type not in ("cpu",
"meta", "disk")`). Per review feedback, this is too permissive — it would silently
promote any unknown device string to accelerator status without validation.

Instead, use a **capability-based check** that verifies PyTorch actually
recognizes the device as a hardware accelerator:

```python
def is_accelerator_type(device_type: str) -> bool:
    """Return True if device_type is a PyTorch-registered hardware accelerator."""
    if device_type in ("cpu", "meta", "disk"):
        return False
    if not torch.accelerator.is_available():
        return False
    return device_type == torch.accelerator.current_accelerator().type
```

This ensures:
- Only the **currently active** accelerator backend is accepted.
- Unregistered or misspelled device strings are rejected.
- No hard-coded allowlist of backend names that would need manual maintenance.
- Backends that register with PyTorch but are not the active accelerator are not
  silently promoted.

## Production Changes

### 1. `cache/base.py` — `cls_from_device()`

Replace explicit `"cuda"` match arms with capability-based guard:

```python
match (device_type, distributed):
    case ("cpu", False):    return CPUCache
    case ("cpu", True):     return DistributedCPUCache
    case ("disk", False):   return DiskCache
    case ("disk", True):    return DistributedDiskCache
    case (accel, False) if is_accelerator_type(accel): return DeviceCache
    case (accel, True) if is_accelerator_type(accel):  return DistributedDeviceCache
    case _: raise NotImplementedError(...)
```

### 2. `cache/disk.py` — `_get_safe_open_device()`

```python
if is_accelerator_type(device.type):
    return torch.accelerator.current_device_index() if device.index is None else device.index
return device.type
```

### 3. `dispatch.py` — `get_device_memory()`

```python
if not torch.accelerator.is_available():
    return {torch.device("cpu"): os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")}

accel_type = torch.accelerator.current_accelerator().type
return {
    torch.device(accel_type, idx): torch.accelerator.get_memory_info(idx)[1]
    for idx in range(torch.accelerator.device_count())
}
```

### 4. `load.py` — `_get_device_memory()`

Same pattern — replace `torch.cuda.get_device_properties(i).total_memory` with
`torch.accelerator.get_memory_info(i)[1]`.

### 5. `dist_utils.py` — `init_dist()`

```python
accel_type = torch.accelerator.current_accelerator().type
device = torch.device(accel_type, local_rank)
torch.accelerator.set_device_index(local_rank)
backend = dist.get_default_backend_for_device(device)
dist.init_process_group(backend=backend, ...)
```

### 6. `convert/helpers.py` — `norm_device()`

Replace `device.type == "cuda"` with `is_accelerator_type(device.type)`.

## Test Harness Migration

This is a first-class workstream, not a mechanical tail. The existing test
infrastructure is CUDA-specific in multiple places that must be migrated together.

### Files Requiring Changes

| File | Current CUDA usage | Required change |
|------|-------------------|-----------------|
| `tests/testing_utils.py` | `torch.cuda.device_count()` in `requires_gpu` | Use `torch.accelerator.is_available()` / `torch.accelerator.device_count()` |
| `tests/test_offload/conftest.py` | `torch.cuda.current_device()`, `torch.cuda.set_device()`, `backend="nccl"` | Use `torch.accelerator` equivalents, backend from `dist.get_default_backend_for_device()` |
| `tests/test_offload/test_dispatch.py` | `patch("...torch.cuda")` mock objects | Patch `torch.accelerator` instead; update mock return shapes |
| `tests/test_compressors/test_compress_decompress_module.py` | `torch.cuda.is_available()` guard | Use `torch.accelerator.is_available()` |
| 10+ offload test files | `torch.device("cuda")` / `cuda_device` fixture | `accel_device` fixture → `torch.accelerator.current_accelerator()` |

### Migration Strategy

1. **Introduce `accel_device` fixture** in `conftest.py`:
   ```python
   @pytest.fixture
   def accel_device():
       return torch.accelerator.current_accelerator()
   ```
2. **Update `requires_gpu`** decorator to use `torch.accelerator.is_available()`.
3. **Replace `cuda_device`** references across all test files.
4. **Update mock targets** in `test_dispatch.py` to patch `torch.accelerator`
   methods instead of `torch.cuda` attributes.
5. **Validate** that all existing CUDA CI tests continue to pass unchanged.

## Safetensors Device String Handling

The `safetensors` library needs to accept the device string when loading tensors
directly to accelerator memory. This is **intended behavior with a defined
fallback**, not just a risk:

### Verified Support

| Backend | `safe_open(device=...)` | Verified version |
|---------|------------------------|------------------|
| CUDA    | `"cuda:0"` — works     | safetensors ≥0.3 |
| XPU     | `"xpu:0"` — works      | safetensors 0.7.0 |

### Fallback Path (for unverified backends)

`_get_safe_open_device()` computes the device argument, but it cannot handle
rejection by `safe_open()` itself — that failure happens later in
`DiskCache.onload()` when the file is actually opened. Therefore, the CPU
fallback must live at the `safe_open()` call site:

```python
def onload(self, offloaded: torch.Tensor | None) -> torch.Tensor | None:
    if offloaded is None:
        return None

    weight_info = self.index[offloaded]
    device = _get_safe_open_device(self.onload_device)

    try:
        with safe_open(
            weight_info["safetensors_file"], framework="pt", device=device
        ) as file:
            onloaded = file.get_tensor(weight_info["weight_name"])
    except (ValueError, RuntimeError):
        # Backend's device string not supported by safetensors — fall back
        # to CPU load, then move to the target device.
        with safe_open(
            weight_info["safetensors_file"], framework="pt", device="cpu"
        ) as file:
            onloaded = file.get_tensor(weight_info["weight_name"])
        onloaded = onloaded.to(self.onload_device)

    onloaded = to_tensor(onloaded, offloaded)
    onloaded = onloaded.to(getattr(torch, weight_info["dtype"]))
    return onloaded
```

`_get_safe_open_device()` itself remains a simple conversion helper with no
try/except — it computes the best-effort device argument. The retry logic is
confined to `onload()` where it can properly re-open the file on CPU.

**Scope decision:** This fallback is added for Tier 1 backends (CUDA, XPU)
where `safe_open` device support is already verified. For Tier 2 backends
where `safe_open` may reject the device string, the fallback provides graceful
degradation rather than a crash. If a Tier 2 backend proves problematic beyond
safetensors compatibility, that is treated as a separate bug.

### Test Coverage

- Add a parametrized test that verifies `safe_open` round-trip for each Tier 1
  backend's device string.
- Add a test that verifies the CPU-fallback path in `DiskCache.onload()` activates
  when `safe_open` rejects an unrecognized device string (mock `safe_open` to
  raise `ValueError` on first call).

## Distributed Backend Design

### Current Behavior

`init_dist()` hardcodes `backend="nccl"`. This works only for CUDA.

### Proposed Behavior

Use PyTorch's public API for backend resolution:
```python
backend = dist.get_default_backend_for_device(torch.device(accel_type))
```

This is preferred over direct indexing into `dist.Backend.default_device_backend_map`
because `get_default_backend_for_device()` is the public contract for this
decision. If the RFC's goal is "use PyTorch's backend registration model," we
should lean on the public API that expresses that contract directly.

### Backend Maturity and Error Handling

Not all backends are equally mature for the collectives used by the offload
distributed cache (`dist.broadcast`, `dist.barrier`, `dist.all_reduce`):

| Backend | Collective lib | Maturity |
|---------|---------------|----------|
| CUDA    | NCCL          | Production-ready |
| XPU     | XCCL (oneCCL) | Stable for common ops; validated in Intel PyTorch CI |
| NPU     | HCCL          | Vendor-maintained; not independently validated |
| HPU     | HCCL (Habana) | Vendor-maintained; not independently validated |

**Design decision:** We do **not** add try/except guards around `init_dist()`.
If a backend's collective library is broken, the failure should surface
immediately and clearly — wrapping it would mask the root cause. Instead:

- `init_dist()` raises `RuntimeError` with a clear message if
  `get_default_backend_for_device()` cannot resolve a backend for the
  active accelerator type.
- Collective call failures propagate as-is from PyTorch — the error messages
  from NCCL/XCCL/HCCL are informative enough for debugging.

### Test Strategy for Distributed

- Existing NCCL-based distributed tests run unchanged on CUDA CI.
- XPU distributed tests are added in a separate test file gated by
  `requires_gpu` + accelerator type check.
- Tier 2 backends: no distributed tests at merge time; added when CI
  infrastructure becomes available.

## Implementation Plan

### Phase 1: Production Code (6 files, ~35 lines)

1. Add `is_accelerator_type()` helper to `offload/__init__.py` or a shared utils
   module.
2. Update `cache/base.py`, `cache/disk.py`, `dispatch.py`, `load.py`,
   `dist_utils.py`, `convert/helpers.py` as described above.
3. Update `setup.py` to declare `torch>=2.6.0`.

### Phase 2: Test Harness Migration (~14 files, ~40 lines)

1. Update `testing_utils.py` — `requires_gpu` decorator.
2. Update `conftest.py` — fixtures and distributed setup.
3. Update `test_dispatch.py` — mock targets.
4. Mechanical fixture renames across remaining offload test files.

### Phase 3: Validation

Run the following matrix before merge:

| Scenario | CUDA | XPU |
|----------|------|-----|
| Single-device offload (CPU↔accel) | ✓ | ✓ |
| Disk offload (disk↔accel) | ✓ | ✓ |
| `dispatch_model` multi-device | ✓ | ✓ |
| `safetensors` direct-load | ✓ | ✓ |
| Distributed init + broadcast | ✓ | ✓ (if multi-XPU available) |
| Full quantization pipeline (W4A16) | ✓ | ✓ |

## Risks

- **PyTorch version bump:** Explicitly handled in Support Policy above.
  `torch>=2.6.0` is declared in `setup.py`.
- **safetensors device strings:** Handled with defined fallback path and test
  coverage (see Safetensors section above).
- **Distributed maturity:** Handled with clear backend maturity table and
  no-silence-errors design (see Distributed section above).
