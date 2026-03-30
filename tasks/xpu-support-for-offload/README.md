# XPU Support for the Offload Module

## Goal

Extend `compressed_tensors.offload` to support Intel XPU devices alongside NVIDIA
CUDA devices. Today the offload system hardcodes `torch.cuda` API calls, the `"cuda"`
device-type string, and the `"nccl"` distributed backend in several places. These
assumptions prevent the module from working on systems with Intel GPUs.

The maintainers have already acknowledged this gap — `dispatch.py` and `load.py` both
contain `# TODO: extend to xpu, ect.` comments, and two test files (`test_module.py`,
`test_dispatch.py`) already reference `torch.device("xpu")`.

---

## Scope

This task covers the `src/compressed_tensors/offload/` package and its corresponding
tests in `tests/test_offload/`. It does **not** cover CUDA references elsewhere in the
repository (e.g. `examples/`, `tests/testing_utils.py`, `tests/test_transform/`).

---

## Current State of XPU References

| Location | What exists |
|----------|-------------|
| `src/compressed_tensors/offload/dispatch.py:246` | `# TODO: extend to xpu, ect.` |
| `src/compressed_tensors/offload/load.py:90` | `# TODO: extend to xpu, ect.` |
| `tests/test_offload/test_module.py:17` | `ONLOAD_DEVICE = torch.device("xpu")` |
| `tests/test_offload/test_dispatch.py:148` | `torch.device("xpu:0")` in `test_dispatch_offloaded` |
| `tests/test_offload/test_module.py:43` | `@requires_gpu` commented out |
| `tests/test_offload/test_module.py:47` | Stray `breakpoint()` left in test |

---

## Files That Require Changes

### Production code — 6 files

#### 1. `src/compressed_tensors/offload/cache/base.py` — **Critical**

**Problem:** `cls_from_device()` (lines 69–86) uses a `match` statement that only
recognises `"cpu"`, `"cuda"`, and `"disk"`. Any other device type (including `"xpu"`)
falls through to `case _:` and raises `NotImplementedError`.

```python
# Current code (lines 69–86)
match (device_type, distributed):
    case ("cpu", False):
        return CPUCache
    case ("cpu", True):
        return DistributedCPUCache
    case ("cuda", False):
        return DeviceCache              # ← works fine for XPU
    case ("cuda", True):
        return DistributedDeviceCache   # ← works fine for XPU
    case ("disk", False):
        return DiskCache
    case ("disk", True):
        return DistributedDiskCache
    case _:
        raise NotImplementedError(...)  # ← "xpu" lands here
```

**Why this is critical:** `DeviceCache` and `DistributedDeviceCache` are fully
device-agnostic — they accept any `onload_device` and use `tensor.to()`. The only
reason they don't work with XPU is this match-statement gatekeeper.

**Required change:** Route any accelerator device type (not just `"cuda"`) to
`DeviceCache` / `DistributedDeviceCache`. This can be done by making those the default
cases, or by checking for `"cpu"` and `"disk"` first and treating everything else as a
device cache.

---

#### 2. `src/compressed_tensors/offload/cache/disk.py` — **High**

**Problem:** `_get_safe_open_device()` (lines 170–185) only special-cases `"cuda"` when
converting a `torch.device` to the format `safetensors.safe_open` expects.

```python
# Current code (lines 178–185)
device = torch.device(device)
if device.type in ("cuda"):
    if device.index is None:
        return torch.cuda.current_device()
    else:
        return device.index
else:
    return device.type              # "xpu:0" → "xpu" (wrong — loses index)
```

For any non-CUDA indexed accelerator (e.g. `"xpu:0"`), this returns the string `"xpu"`
instead of the integer index `0`, which will likely fail or return data on the wrong
device.

**Required change:** Generalise the index-resolution logic to handle any accelerator
with indexed devices. At minimum, add `"xpu"` to the type check. Better: treat any
device with a non-None `.index` as an indexed accelerator. Also replace
`torch.cuda.current_device()` with a device-aware equivalent for the bare-device case.

---

#### 3. `src/compressed_tensors/offload/dispatch.py` — **High**

**Problem:** `get_device_memory()` (lines 227–249) only discovers CUDA devices.

```python
# Current code (lines 234–249)
if not torch.cuda.is_available():
    ...  # CPU fallback

if dist.is_available() and dist.is_initialized():
    device_memory = torch.cuda.get_device_properties(dist.get_rank()).total_memory
    return {torch.device("cuda"): device_memory}

return {
    torch.device(f"cuda:{idx}"): torch.cuda.get_device_properties(idx).total_memory
    for idx in range(torch.cuda.device_count())
}
```

**Hardcoded CUDA API calls (5):**

| Line | API call |
|------|----------|
| 234 | `torch.cuda.is_available()` |
| 242 | `torch.cuda.get_device_properties(dist.get_rank()).total_memory` |
| 243 | `torch.device("cuda")` |
| 247 | `torch.device(f"cuda:{idx}")`, `torch.cuda.get_device_properties(idx).total_memory` |
| 248 | `torch.cuda.device_count()` |

**Required change:** Add equivalent XPU discovery using `torch.xpu.is_available()`,
`torch.xpu.device_count()`, and `torch.xpu.get_device_properties(idx).total_memory`.
The function should try CUDA first, then XPU, then fall back to CPU. Also fix the
log message on line 241 (`"Dispatching to local rank gpu"` — should be device-agnostic).

---

#### 4. `src/compressed_tensors/offload/load.py` — **High**

**Problem:** `_get_device_memory()` (lines 89–98) is a second, independent
device-discovery function with the same CUDA-only limitation.

```python
# Current code (lines 89–98)
def _get_device_memory() -> dict[int, int]:
    # TODO: extend to xpu, ect.
    if is_distributed():
        index = dist.get_rank()
        return {index: torch.cuda.get_device_properties(index).total_memory}
    else:
        return {
            index: torch.cuda.get_device_properties(index).total_memory
            for index in range(torch.cuda.device_count())
        }
```

**Hardcoded CUDA API calls (3):**

| Line | API call |
|------|----------|
| 93 | `torch.cuda.get_device_properties(index).total_memory` |
| 96 | `torch.cuda.get_device_properties(index).total_memory` |
| 97 | `torch.cuda.device_count()` |

**Required change:** Mirror the fix applied to `dispatch.py:get_device_memory()`. Both
functions should use the same device-discovery logic (consider extracting a shared
helper).

---

#### 5. `src/compressed_tensors/offload/dist_utils.py` — **High**

**Problem:** `init_dist()` (lines 21–41) hardcodes CUDA device construction and the
NCCL backend.

```python
# Current code (lines 32–35)
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
dist.init_process_group(
    backend="nccl",
    ...
)
```

**Hardcoded CUDA items (3):**

| Line | Item |
|------|------|
| 32 | `torch.device(f"cuda:{local_rank}")` |
| 33 | `torch.cuda.set_device(device)` |
| 35 | `backend="nccl"` |

**Required change:** Detect the available accelerator and choose the appropriate device
constructor and backend:

| Accelerator | Device | `set_device` | Backend |
|-------------|--------|-------------|---------|
| CUDA | `cuda:{rank}` | `torch.cuda.set_device()` | `nccl` |
| XPU | `xpu:{rank}` | `torch.xpu.set_device()` | `ccl` |

Also update the `as_broadcastable()` docstring (lines 55–57) which references
NVIDIA-specific `sm_90` / Hopper hardware. The FP8-to-uint8 workaround itself is
backend-agnostic and likely still needed for oneCCL.

---

#### 6. `src/compressed_tensors/offload/convert/helpers.py` — **Medium**

**Problem:** `norm_device()` (lines 18–40) only normalises bare `"cuda"` to `"cuda:0"`.

```python
# Current code (line 37)
if not is_distributed() and device.type == "cuda" and device.index is None:
    device = torch.device(type=device.type, index=0)
```

A bare `"xpu"` device would not be normalised to `"xpu:0"`, potentially causing
inconsistencies downstream.

**Required change:** Remove the `device.type == "cuda"` guard. Any non-distributed
accelerator device with `index is None` should be normalised to index 0.

---

### Test code — 14 files

#### `tests/testing_utils.py`

`requires_gpu` (lines 114–142) and `is_gpu_available` (lines 102–111) only check
`torch.cuda.device_count()`. These should additionally check `torch.xpu.device_count()`
so that XPU-only machines are not unconditionally skipped.

#### `tests/test_offload/conftest.py`

- `assert_device_equal` (line 26) calls `torch.cuda.current_device()` unconditionally —
  will crash on machines without CUDA. Needs a device-type check or a try/except.
- `distributed_fixture` (line 75) calls `torch.cuda.set_device()` and uses
  `backend="nccl"`. Needs the same CUDA/XPU branching as `dist_utils.init_dist()`.
- `cuda_device` fixture (lines 109–114) always returns `torch.device("cuda")`. Consider
  renaming to `accelerator_device` and returning the available accelerator.

#### `tests/test_offload/test_module.py`

- Uses `ONLOAD_DEVICE = torch.device("xpu")` (line 17) — already targets XPU but
  `@requires_gpu` is commented out and there is a stray `breakpoint()` (line 47). These
  need cleanup.

#### `tests/test_offload/test_dispatch.py`

- `test_dispatch_offloaded` (line 145) uses `xpu:0` with `@requires_gpu` commented out.
  All other tests use hardcoded `"cuda:0"` / `"cuda:1"`.

#### `tests/test_offload/test_interface.py`

- `ONLOAD_DEVICE = torch.device("cuda")` (line 21). Needs to support XPU.

#### `tests/test_offload/test_load.py`

- Parametrised with `torch.device("cuda")` (lines 28–35). Needs XPU variants.

#### Cache test files (6 files)

Every cache test hardcodes `torch.device("cuda")` in its `onload_device` fixture:

| File | Line |
|------|------|
| `tests/test_offload/cache/test_cpu.py` | 22 |
| `tests/test_offload/cache/test_disk.py` | 28 |
| `tests/test_offload/cache/test_device.py` | 25 |
| `tests/test_offload/cache/test_dist_cpu.py` | 25 |
| `tests/test_offload/cache/test_dist_device.py` | 31, 210 |
| `tests/test_offload/cache/test_dist_disk.py` | 30, 210 |

#### Convert test files (3 files)

All use the `cuda_device` fixture and hardcode `"cuda"` strings:

| File | Lines |
|------|-------|
| `tests/test_offload/convert/test_convert.py` | 31, 48–49, 52 |
| `tests/test_offload/convert/test_from_accelerate.py` | 34, 53, 81 |
| `tests/test_offload/convert/test_to_accelerate.py` | 19, 25, 27, 30–33, 46–48, 53 |

---

## `torch.cuda` → `torch.xpu` API Mapping

| CUDA API | XPU Equivalent |
|----------|----------------|
| `torch.cuda.is_available()` | `torch.xpu.is_available()` |
| `torch.cuda.device_count()` | `torch.xpu.device_count()` |
| `torch.cuda.get_device_properties(i).total_memory` | `torch.xpu.get_device_properties(i).total_memory` |
| `torch.cuda.current_device()` | `torch.xpu.current_device()` |
| `torch.cuda.set_device(d)` | `torch.xpu.set_device(d)` |
| `backend="nccl"` | `backend="ccl"` (Intel oneCCL) |
| `.cuda()` | `.to("xpu")` |

---

## Files That Are Already Device-Agnostic (No Changes Needed)

These files work through the `OffloadCache` abstraction or `tensor.to()` and contain no
CUDA references:

| File | Notes |
|------|-------|
| `src/compressed_tensors/offload/module.py` | All device handling via parameters |
| `src/compressed_tensors/offload/utils.py` | Uses `tensor.to()`, CPU-only fallback |
| `src/compressed_tensors/offload/cache/cpu.py` | Offload target is always CPU |
| `src/compressed_tensors/offload/cache/device.py` | Fully generic — accepts any device |
| `src/compressed_tensors/offload/cache/dist_cpu.py` | Uses `dist.broadcast_object_list` (backend-agnostic) |
| `src/compressed_tensors/offload/cache/dist_device.py` | Uses `dist.broadcast` (backend-agnostic) |
| `src/compressed_tensors/offload/cache/dist_disk.py` | Uses `dist.broadcast_object_list` (backend-agnostic) |
| `src/compressed_tensors/offload/cache/__init__.py` | Pure re-exports |
| `src/compressed_tensors/offload/convert/__init__.py` | Pure re-exports |
| `src/compressed_tensors/offload/convert/from_accelerate.py` | Reads `.device.type` dynamically |
| `src/compressed_tensors/offload/convert/to_accelerate.py` | Delegates to `norm_device` |

---

## Suggested Implementation Approach

### Step 1: Add a device-detection helper

Create a small utility (e.g. in `dist_utils.py` or a new `device_utils.py`) that
abstracts accelerator detection:

```python
def get_accelerator_type() -> str | None:
    """Return "cuda", "xpu", or None."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return None

def accelerator_device_count() -> int:
    acc = get_accelerator_type()
    if acc == "cuda":
        return torch.cuda.device_count()
    elif acc == "xpu":
        return torch.xpu.device_count()
    return 0

def get_accelerator_total_memory(index: int) -> int:
    acc = get_accelerator_type()
    if acc == "cuda":
        return torch.cuda.get_device_properties(index).total_memory
    elif acc == "xpu":
        return torch.xpu.get_device_properties(index).total_memory
    raise RuntimeError(f"No accelerator available for index {index}")

def set_accelerator_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "xpu":
        torch.xpu.set_device(device)

def get_current_accelerator_device_index() -> int:
    acc = get_accelerator_type()
    if acc == "cuda":
        return torch.cuda.current_device()
    elif acc == "xpu":
        return torch.xpu.current_device()
    raise RuntimeError("No accelerator available")

def get_dist_backend() -> str:
    acc = get_accelerator_type()
    if acc == "xpu":
        return "ccl"
    return "nccl"
```

### Step 2: Update production code (6 files)

Replace all `torch.cuda.*` calls with the helpers from Step 1. For each file:

1. **`cache/base.py`** — Replace the `("cuda", ...)` match arms with a catch-all for
   non-cpu, non-disk device types.
2. **`cache/disk.py`** — Generalise `_get_safe_open_device()` to handle any indexed
   accelerator.
3. **`dispatch.py`** — Rewrite `get_device_memory()` using the helpers.
4. **`load.py`** — Rewrite `_get_device_memory()` using the helpers (or share the same
   function with `dispatch.py`).
5. **`dist_utils.py`** — Rewrite `init_dist()` using the helpers.
6. **`convert/helpers.py`** — Generalise `norm_device()` to normalise any accelerator,
   not just CUDA.

### Step 3: Update test infrastructure

1. **`tests/testing_utils.py`** — Extend `requires_gpu` / `is_gpu_available` to also
   detect XPU devices.
2. **`tests/test_offload/conftest.py`** — Fix `assert_device_equal` to not call
   `torch.cuda.current_device()` unconditionally. Rename `cuda_device` fixture to
   something device-agnostic.

### Step 4: Parametrise tests for device type

Instead of hardcoding `torch.device("cuda")` in every `onload_device` fixture, use a
shared fixture or parametrisation that yields the available accelerator device. This
covers all 14 test files listed above.

### Step 5: Clean up existing XPU work-in-progress

- `test_module.py:47` — Remove stray `breakpoint()`.
- `test_module.py:43`, `test_dispatch.py:144` — Re-enable `@requires_gpu` once XPU
  detection works.
- Remove the `# TODO: extend to xpu, ect.` comments in `dispatch.py` and `load.py`.

---

## Testing Strategy

- **Unit tests without a device:** The cache base class routing (`cls_from_device`) and
  `norm_device` can be tested with mock device strings — no hardware needed.
- **Unit tests with mocked `torch.xpu`:** Device discovery in `get_device_memory()` and
  `_get_device_memory()` can be tested by patching `torch.xpu.is_available()` etc.,
  similar to the existing `test_cpu_dispatch` tests that mock `torch.cuda`.
- **Integration tests:** Require an Intel XPU. Guard with `@requires_gpu` once it
  detects XPU. The existing `test_dispatch_offloaded` (which already uses `xpu:0`) is a
  starting point.
- **CI:** Requires a runner with Intel XPU hardware. Until that is available, XPU paths
  should be tested locally and the mock-based unit tests should run in existing CI.
