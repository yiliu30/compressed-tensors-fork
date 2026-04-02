# Fake XPU Testing Strategy for `torch.accelerator` Migration

> **Status**: Proposal — for discussion in [#655](https://github.com/vllm-project/compressed-tensors/issues/655)
> **Date**: 2026-04-02
> **Author**: Yi Liu

## Problem

The `torch.accelerator` migration replaces all `torch.cuda.*` calls with device-agnostic APIs.
CUDA CI validates tensor operations, but we need confidence that XPU code paths work correctly
**before real XPU CI is available** (Buildkite setup in progress).

Brian asked: *"Can you explain how you would handle the fake XPU testing device? A mock-up in tests/?"*

## Background

### The C++ Barrier

`tensor.to(torch.device("xpu", 0))` dispatches to the XPU backend at the C++ level.
No Python-level mock can make this work without real XPU hardware.
Similarly, `torch.randn(device="xpu:0")` fails at C++ before any Python interception.
`tensor.device` is a C-level read-only property.

This fundamental constraint limits what "fake XPU" can mean — and shapes all options below.

### Current Test Architecture

- **109 test functions** in `tests/test_offload/`
- **102** use `@requires_gpu` — these move tensors to the accelerator device
- **~16 files** set module-level constants at import time:
  ```python
  ONLOAD_DEVICE = torch.device(torch.accelerator.current_accelerator().type)
  _ACCEL_TYPE = torch.accelerator.current_accelerator().type
  ```
- **Device assertions**: all go through `assert_device_equal()` in conftest.py (raw `.type` string comparison)
  or `torch.device(a) == torch.device(b)` (interceptable by TorchFunctionMode)
- **Production code** reads `torch.accelerator` at **call time**, not import time

### Other Approaches Investigated and Ruled Out

| Approach | Why Not |
|---|---|
| **FakeTensorMode** | Only works if target backend is compiled in the PyTorch build — useless for cross-device simulation. Also can't support `torch.equal`, `Module.to`, or value reads. |
| **privateuse1 backend registration** | Provides naming (`torch.utils.rename_privateuse1_backend`) but tensor allocation needs a C++ backend module. Writing a real allocator is massive effort for a test-only purpose. |
| **conftest early-patch only** (no tensor layer) | Patches accelerator before imports so module constants get "xpu", but `tensor.to("xpu:0")` still fails at C++. Breaks existing non-GPU tests. Adds zero XPU coverage over Option 1. |
| **Tensor.to monkey-patch** | Can remap device strings in `tensor.to()`, but `tensor.device` still reports the real device. Assertions comparing `tensor.device.type` against expected device type fail. |

---

## Viable Options

### Option 1: Mock Routing Tests (Focused Unit Tests)

Add a dedicated `test_xpu_routing.py` that mocks `torch.accelerator` to report `"xpu"`, then
calls production functions directly to verify branching logic. No tensor operations.

```python
# tests/test_offload/test_xpu_routing.py
from types import SimpleNamespace
import pytest
import torch
from compressed_tensors.offload.convert.helpers import is_accelerator_type, norm_device

@pytest.fixture
def mock_xpu_accelerator(monkeypatch):
    """Mock torch.accelerator to report XPU as the current device."""
    fake = SimpleNamespace(type="xpu")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: fake)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 1)

def test_is_accelerator_type_xpu(mock_xpu_accelerator):
    assert is_accelerator_type("xpu") is True
    assert is_accelerator_type("cuda") is False

def test_cache_routes_for_xpu(mock_xpu_accelerator):
    from compressed_tensors.offload.cache.base import OffloadCache
    from compressed_tensors.offload.cache.device import DeviceCache
    CacheClass = OffloadCache.get("xpu")
    assert CacheClass is DeviceCache

def test_norm_device_resolves_xpu(mock_xpu_accelerator):
    assert norm_device("xpu") == torch.device("xpu", 0)
```

**What it tests**: `is_accelerator_type()`, cache routing, `norm_device()`, safetensors device mapping, dist backend selection.

**What it doesn't test**: Actual tensor creation/movement under a fake XPU identity.

| Pros | Cons |
|---|---|
| Self-contained, easy to review | Only tests routing logic, not tensor layer |
| No risk to existing tests | |
| Matches existing pattern (`test_dispatch_cpu_only_via_fallback`) | |
| Works on any CI machine (CUDA, CPU-only, XPU) | |
| Low effort (~60 lines, 1 file) | |

---

### Option 2: TorchFunctionMode Tensor Layer Emulation

Use PyTorch's [`TorchFunctionMode`](https://pytorch.org/docs/stable/torch.overrides.html) (public API) to intercept
**all torch function calls** and transparently remap device strings at the Python level.
Combined with an early conftest patch, this allows the **existing 102 GPU tests** to run end-to-end
on CUDA hardware while the code path uses XPU device strings.

#### How it works

`TorchFunctionMode` intercepts every `torch.*` function, including:
- `torch.device("xpu", 0)` → remapped to `torch.device("cuda", 0)`
- `tensor.to("xpu:0")` → remapped to `tensor.to("cuda:0")`
- `torch.randn(device="xpu:0")` → remapped to `torch.randn(device="cuda:0")`
- `torch.nn.Linear(device="xpu:0")` → remapped to device `"cuda:0"`
- `torch.equal(a, b)` → works normally (both tensors on real CUDA)

```python
from torch.overrides import TorchFunctionMode

class DeviceRemapMode(TorchFunctionMode):
    """Transparently remap device type strings in all torch operations.

    When activated, any torch function receiving a device argument with
    ``fake_type`` will have it silently replaced with ``real_type`` before
    the call reaches the C++ backend.
    """
    def __init__(self, fake_type: str, real_type: str):
        self.fake_type = fake_type  # e.g. "xpu" (what code thinks)
        self.real_type = real_type  # e.g. "cuda" (what hardware is)

    def _remap(self, arg):
        if isinstance(arg, torch.device) and arg.type == self.fake_type:
            return torch.device(self.real_type, arg.index)
        if isinstance(arg, str) and self.fake_type in arg:
            return arg.replace(self.fake_type, self.real_type)
        return arg

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        new_args = tuple(self._remap(a) for a in args)
        new_kwargs = {k: self._remap(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
```

#### Conftest plugin design

```python
# conftest.py or pytest plugin
def pytest_addoption(parser):
    parser.addoption("--emulate-xpu", action="store_true", default=False)

def pytest_configure(config):
    if config.getoption("--emulate-xpu"):
        real_type = torch.accelerator.current_accelerator().type  # "cuda"
        fake_type = "xpu"
        config._orig_current_accelerator = torch.accelerator.current_accelerator

        # Activate DeviceRemapMode BEFORE test module imports
        # This intercepts torch.device() in module-level constants
        mode = DeviceRemapMode(fake_type=fake_type, real_type=real_type)
        mode.__enter__()
        config._device_remap_mode = mode

        # Mock accelerator to report fake identity
        fake_accel = SimpleNamespace(type=fake_type)
        torch.accelerator.current_accelerator = lambda: fake_accel

def pytest_unconfigure(config):
    mode = getattr(config, "_device_remap_mode", None)
    if mode is not None:
        mode.__exit__(None, None, None)
    orig = getattr(config, "_orig_current_accelerator", None)
    if orig is not None:
        torch.accelerator.current_accelerator = orig
```

#### What happens at each stage

| Stage | What code does | What actually happens |
|---|---|---|
| Module import | `_ACCEL_TYPE = torch.accelerator.current_accelerator().type` | Gets `"xpu"` (mocked) |
| Module import | `ONLOAD_DEVICE = torch.device(_ACCEL_TYPE)` | `torch.device("xpu")` intercepted → becomes `torch.device("cuda")` |
| Test execution | `tensor.to(ONLOAD_DEVICE)` | `ONLOAD_DEVICE` is already `cuda` → lands on real CUDA ✓ |
| Test execution | `tensor.to("xpu:0")` (from f-string `_DEV0`) | Intercepted → remapped to `"cuda:0"` ✓ |
| Test execution | `torch.randn(device="xpu:0")` | Intercepted → allocated on CUDA ✓ |
| Assertions | `torch.device(a) == torch.device(b)` | Both sides go through `torch.device()` → remapped consistently ✓ |
| Assertions | `assert_device_equal(tensor.device, ONLOAD_DEVICE)` | Both are `"cuda"` (remapped) → ✓ |

#### Verified end-to-end (on CUDA hardware, emulating XPU identity)

```
offload_module(linear, ONLOAD_DEVICE, OFFLOAD_DEVICE)  ✓
linear.weight.device matches ONLOAD_DEVICE              ✓
torch.equal(onloaded_weight, original_weight)            ✓
linear(input)  # forward pass                            ✓
isinstance(linear._parameters, CPUCache)                 ✓
assert_device_equal(weight.device, ONLOAD_DEVICE)        ✓
```

#### Known limitation: `is_accelerator_type` mismatch

Production code does:
```python
device_type = torch.device(device).type  # remapped to "cuda"
is_accelerator_type(device_type)         # "cuda" == "xpu" (mocked) → False!
```

The device object is remapped to `"cuda"`, but `is_accelerator_type` compares against the mocked
accelerator type `"xpu"`. **Fix**: patch `is_accelerator_type` to accept both types when emulating:

```python
original_is_accel = is_accelerator_type
def patched_is_accel(device_type):
    return device_type in (fake_type, real_type)
```

This is a trade-off: we bypass the routing function to let end-to-end tests pass. Routing logic
must still be tested separately (via Option 1).

| Pros | Cons |
|---|---|
| Runs **all 102 GPU tests** under fake XPU identity | Requires `is_accelerator_type` patch (bypasses routing) |
| Real tensor ops on real hardware | 3 layers of patching: accelerator mock + TorchFunctionMode + `is_accelerator_type` |
| End-to-end pipeline validation | `TorchFunctionMode` behavior may shift across PyTorch versions |
| Catches integration bugs that routing-only tests miss | CI needs `--emulate-xpu` flag |
| | `torch.accelerator.*` memory/sync functions go to real CUDA |
| | `dist.get_default_backend_for_device` returns NCCL (not XCCL) |

---

## Recommended Strategy: Option 1 + Option 2 Combined

Use **both** approaches for layered coverage:

| Layer | Approach | What it validates |
|---|---|---|
| **Routing logic** | Option 1: mock routing tests | `is_accelerator_type`, `norm_device`, cache selection, dist backend — the code paths that actually changed |
| **Tensor pipeline** | Option 2: TorchFunctionMode emulation | End-to-end offload/onload/forward under fake XPU identity — catches integration bugs |
| **Real hardware** | Neither (requires real XPU CI) | XPU kernels, XCCL backend, memory semantics, `safetensors` device recognition, op coverage gaps |

### Coverage at a glance

| Level | What it answers | Covered by |
|---|---|---|
| **Routing logic** — does code branch correctly for `"xpu"`? | Cache selection, device normalization, dist backend | Option 1 |
| **Module constants** — do test files importing `torch.accelerator` at load time get consistent values? | 18 files with `_ACCEL_TYPE`, `ONLOAD_DEVICE` at module scope | Option 2 |
| **Tensor pipeline** — do offload/onload/forward cycles work under non-CUDA device strings? | Tensor creation, movement, `nn.Module`, assertions | Option 2 |
| **Real hardware** — does it work on XPU silicon? | Kernels, XCCL backend, memory semantics, op coverage | Real XPU CI (Phase 3) |

**Key trade-off**: Option 2 must patch `is_accelerator_type` to accept both `"cuda"` and
`"xpu"`, which **bypasses routing logic**. This is why both options are needed together —
Option 1 covers routing, Option 2 covers the tensor pipeline, neither alone is sufficient.

### CI configuration

```yaml
# Normal run (every PR):
pytest tests/

# Emulated XPU run (nightly or on-demand):
pytest tests/ --emulate-xpu
```

### What each layer catches

| Bug type | Option 1 | Option 2 | Real XPU CI |
|---|:---:|:---:|:---:|
| Wrong cache type for "xpu" | ✅ | ✅ | ✅ |
| `norm_device("xpu")` resolves wrong | ✅ | ✅ | ✅ |
| Wrong dist backend for XPU | ✅ | ❌ (uses NCCL) | ✅ |
| Offload/onload cycle breaks under non-cuda type strings | ❌ | ✅ | ✅ |
| Forward pass fails after offload with non-cuda type | ❌ | ✅ | ✅ |
| safetensors load with non-cuda device string | ❌ | ✅ | ✅ |
| XPU kernel incompatibility | ❌ | ❌ | ✅ |
| XPU memory reporting issues | ❌ | ❌ | ✅ |

### Comparison Matrix

| Criterion | Option 1 only | Option 2 only | Option 1 + 2 |
|---|---|---|---|
| Routing coverage | ✅ | ⚠️ bypassed | ✅ |
| Tensor pipeline coverage | ❌ | ✅ (102 tests) | ✅ (102 tests) |
| Implementation effort | Low | Medium | Medium |
| Maintenance burden | Low | Medium | Medium |
| Risk to existing tests | None | Low (opt-in flag) | Low |
| CI complexity | None | `--emulate-xpu` | `--emulate-xpu` |
| PyTorch version sensitivity | None | Medium | Medium |

---

## Coverage Scope (Detailed)

This section documents exactly what each option covers across the codebase, what falls
through, and why. It also specifies the test harness changes required.

### Test landscape

| Directory | Test functions | GPU tests | Module-level device constants |
|---|---:|---:|---|
| `tests/test_offload/` | 109 | 102 (`@requires_gpu`) | 11 files: `ONLOAD_DEVICE`, `_ACCEL_TYPE`, `_DEV0` |
| `tests/test_transform/` | 22 | ~18 | 3 files: `_ACCEL_TYPE` |
| `tests/test_compressors/` | 40 | ~5 | 1 file |
| `tests/test_quantization/` | 63 | ~4 | 1 file (`test_initialize.py`) |
| `tests/test_modeling/` | 2 | 2 | 2 files |
| Other (`test_configs`, `test_utils`, etc.) | 25 | 0 | 0 |
| **Total** | **261** | **~131** | **18 files** |

### Option 1 coverage scope: routing logic

Option 1 targets the **decision functions** that changed in the `torch.accelerator` migration.
These are pure-logic functions with no tensor operations — perfect for mock-based testing.

**Production functions tested:**

| Function | Location | What it decides |
|---|---|---|
| `is_accelerator_type(device_type)` | `offload/convert/helpers.py:23` | `device_type == torch.accelerator.current_accelerator().type` |
| `norm_device(device)` | `offload/convert/helpers.py:39` | Resolves bare `"xpu"` → `torch.device("xpu", 0)` for non-dist |
| `OffloadCache.cls_from_device(device)` | `offload/cache/base.py:46` | Selects cache class: `DeviceCache` / `DistributedDeviceCache` / etc. |
| `_get_safe_open_device(device)` | `offload/cache/disk.py:164` | Converts `torch.device` → `safetensors.safe_open` device arg |
| `init_dist()` | `offload/dist_utils.py:21` | Calls `dist.get_default_backend_for_device(device)` |

**Callers in production code that depend on these:**

- `OffloadCache.cls_from_device()` → called by `offload_module`, `convert_from_accelerate` (every offload path)
- `is_accelerator_type()` → called by `cls_from_device` (lines 72-74), `norm_device` (line 61), `DiskCache.onload` (line 176)
- `_get_safe_open_device()` → called by `DiskCache.onload()` to determine safetensors load device

**What Option 1 does NOT cover:**

- Tensor creation, movement, or computation under non-CUDA device strings
- End-to-end offload → CPU → onload → forward pass pipeline
- Module-level constant resolution (`_ACCEL_TYPE`, `ONLOAD_DEVICE`)
- `safetensors.safe_open(device=...)` with non-CUDA strings
- `assert_device_equal` / `assert_tensor_equal` behavior

### Option 2 coverage scope: tensor pipeline

Option 2 validates that the **entire test suite** runs without hardcoded `"cuda"` assumptions
in the tensor pipeline. It does **not** test XPU-specific behavior — it tests
**device-string agnosticism**.

**What TorchFunctionMode intercepts (verified on PyTorch 2.11):**

| Call pattern | Intercepted? | Behavior |
|---|---|---|
| `torch.device("xpu", 0)` | ✅ | Returns `torch.device("cuda", 0)` |
| `torch.randn(device="xpu:0")` | ✅ | Allocates on CUDA |
| `tensor.to("xpu:0")` | ✅ | Moves to CUDA |
| `tensor.to(torch.device("xpu", 0))` | ✅ | Moves to CUDA |
| `torch.nn.Linear(device="xpu:0")` | ✅ | Parameters on CUDA |
| `torch.equal(a, b)` | ✅ | Both on CUDA, works normally |
| `torch.zeros_like(t)` | ✅ | Inherits CUDA device |
| `safetensors.safe_open(device=N)` | N/A | `_get_safe_open_device` resolves to int index (patched `is_accelerator_type`) |

**What TorchFunctionMode does NOT intercept:**

| Call pattern | Why not | Impact |
|---|---|---|
| `tensor.device` | C-level read-only property | Reports `"cuda"` — OK, both sides of assertions see `"cuda"` |
| `torch.accelerator.current_accelerator()` | Not a torch function | Handled by mock (Layer 1) |
| `torch.accelerator.synchronize()` | Not a torch function | Calls real CUDA sync — semantically different from XPU sync |
| `torch.accelerator.memory_stats()` | Not a torch function | Reports CUDA stats, not XPU |
| `dist.get_default_backend_for_device()` | torch.distributed, not torch function | Returns NCCL, not XCCL |
| C++ extension ops | Below Python dispatch | N/A for this codebase |

**The `is_accelerator_type` paradox (why Option 2 must patch it):**

Under `--emulate-xpu`, device objects are remapped to `"cuda"` before reaching C++.
But `is_accelerator_type` compares `device.type` against the mocked accelerator type:

```
torch.device("xpu", 0)  →  TorchFunctionMode remaps →  torch.device("cuda", 0)
                                                              │
is_accelerator_type("cuda")  →  "cuda" == "xpu" (mocked) → False  ← WRONG
```

This breaks three production paths:
1. `OffloadCache.cls_from_device()` — fails to select `DeviceCache`
2. `norm_device()` — fails to resolve bare accelerator to index 0
3. `DiskCache.onload()` — fails to resolve safetensors device

**Fix**: patch `is_accelerator_type` to accept both fake and real types. This means
Option 2 **bypasses routing logic** — routing coverage comes exclusively from Option 1.

> **Why the module-level patch works**: Both `OffloadCache.cls_from_device()` (base.py:63)
> and `_get_safe_open_device()` (disk.py:173) use **late imports** inside function bodies:
> `from compressed_tensors.offload.convert.helpers import is_accelerator_type`.
> Each call re-reads from the module object, so patching `helpers_mod.is_accelerator_type`
> in `pytest_configure` propagates to all callers without needing per-call-site patches.

**Module-level constants (18 files):**

These read `torch.accelerator.current_accelerator().type` at import time:

```python
# Pattern A: string constant (7 files)
_ACCEL_TYPE = torch.accelerator.current_accelerator().type  # → "xpu" (mocked)

# Pattern B: device object (8 files)
ONLOAD_DEVICE = torch.device(torch.accelerator.current_accelerator().type)
# → torch.device("xpu") → intercepted by TorchFunctionMode → torch.device("cuda", None)

# Pattern C: inline in fixtures/functions (3 files)
accel_type = torch.accelerator.current_accelerator().type  # → "xpu" (mocked)
```

For `--emulate-xpu` to work, the `DeviceRemapMode` and accelerator mock must both be
active **before** `pytest` collects test modules (triggering imports). This is achieved
by activating in `pytest_configure`, which runs before collection.

**Test directories affected by `--emulate-xpu`:**

| Directory | Tests run | Expected outcome |
|---|---|---|
| `tests/test_offload/` | All 109 | ✅ Full pass (all patterns covered) |
| `tests/test_transform/` | All 22 | ✅ Full pass (same constant patterns) |
| `tests/test_compressors/` | All 40 | ✅ Full pass (minimal device dependency) |
| `tests/test_quantization/` | All 63 | ✅ Full pass (1 file with device constants) |
| `tests/test_modeling/` | Both 2 | ✅ Full pass (same patterns) |
| `tests/test_configs/` | All 5 | ✅ No device dependency |
| `tests/test_utils/` | All 17 | ✅ No device dependency |

### Coverage gaps (neither option covers)

| Gap | Why | Mitigation |
|---|---|---|
| XPU kernel dispatch (SYCL vs CUDA kernels) | No XPU hardware / backend compiled | Real XPU CI only |
| XPU memory semantics (`torch.accelerator.memory_stats`) | `torch.accelerator.*` not intercepted | Real XPU CI only |
| XCCL collective backend | `dist.get_default_backend_for_device` returns NCCL | Option 1 can mock-test the call; real validation needs XPU CI |
| `safetensors.safe_open(device="xpu:0")` literal string | safetensors library may not recognize `"xpu"` | Production code already has CPU fallback (`DiskCache.onload` line 63) |
| PyTorch XPU-specific op support gaps | Not detectable without XPU build | Real XPU CI only |

---

## Test Harness Changes

### New files

#### `tests/test_offload/test_xpu_routing.py` (Option 1)

Mock routing tests. No GPU required, runs on any CI machine.

```python
# tests/test_offload/test_xpu_routing.py
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload.cache.base import OffloadCache
from compressed_tensors.offload.cache.device import DeviceCache
from compressed_tensors.offload.convert.helpers import (
    is_accelerator_type,
    norm_device,
)


@pytest.fixture
def mock_xpu_accelerator(monkeypatch):
    """Mock torch.accelerator to report XPU as the current device."""
    fake = SimpleNamespace(type="xpu")
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: fake)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 1)


@pytest.mark.unit
class TestXpuRouting:
    """Verify that routing functions correctly handle 'xpu' as the accelerator type.

    These tests mock torch.accelerator without real tensor operations —
    they validate the decision logic that changed in the torch.accelerator migration.
    """

    def test_is_accelerator_type_xpu(self, mock_xpu_accelerator):
        assert is_accelerator_type("xpu") is True
        assert is_accelerator_type("cuda") is False
        assert is_accelerator_type("cpu") is False

    def test_is_accelerator_type_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        assert is_accelerator_type("xpu") is False

    def test_cache_routes_device_cache_for_xpu(self, mock_xpu_accelerator):
        cache_cls = OffloadCache.cls_from_device(torch.device("xpu", 0))
        assert cache_cls is DeviceCache

    def test_norm_device_resolves_xpu_to_index_0(self, mock_xpu_accelerator):
        result = norm_device("xpu")
        assert result == torch.device("xpu", 0)

    def test_norm_device_preserves_xpu_with_index(self, mock_xpu_accelerator):
        result = norm_device(torch.device("xpu", 0))
        assert result == torch.device("xpu", 0)

    def test_norm_device_cpu_unaffected(self, mock_xpu_accelerator):
        result = norm_device("cpu")
        assert result == torch.device("cpu")

    def test_get_safe_open_device_xpu(self, mock_xpu_accelerator):
        from compressed_tensors.offload.cache.disk import _get_safe_open_device
        # bare "xpu" → current device index (0)
        result = _get_safe_open_device(torch.device("xpu"))
        assert result == 0

    def test_get_safe_open_device_xpu_with_index(self, mock_xpu_accelerator):
        from compressed_tensors.offload.cache.disk import _get_safe_open_device
        result = _get_safe_open_device(torch.device("xpu", 3))
        assert result == 3

    def test_get_safe_open_device_cpu(self, mock_xpu_accelerator):
        from compressed_tensors.offload.cache.disk import _get_safe_open_device
        result = _get_safe_open_device(torch.device("cpu"))
        assert result == "cpu"
```

#### `tests/emulate_device.py` (Option 2 — shared infrastructure)

Reusable device emulation utilities for the pytest plugin and standalone demo.

```python
# tests/emulate_device.py
"""
Device emulation utilities for --emulate-xpu testing.

Provides DeviceRemapMode (TorchFunctionMode subclass) that transparently remaps
device type strings in all torch.* function calls, and a pytest plugin that
activates it before test collection.
"""
import re
from types import SimpleNamespace

import torch
from torch.overrides import TorchFunctionMode


class DeviceRemapMode(TorchFunctionMode):
    """Transparently remap device type strings in all torch operations.

    When activated, any torch function receiving a device argument with
    ``fake_type`` will have it silently replaced with ``real_type`` before
    the call reaches the C++ backend.

    Uses a strict regex pattern to avoid false-positive replacements on
    strings that happen to contain the fake device type as a substring.
    """

    def __init__(self, fake_type: str, real_type: str):
        self.fake_type = fake_type
        self.real_type = real_type
        self._device_pat = re.compile(rf"^{re.escape(fake_type)}(?::\d+)?$")

    def _remap(self, arg):
        if isinstance(arg, torch.device):
            if arg.type == self.fake_type:
                return torch.device(self.real_type, arg.index)
        elif isinstance(arg, str) and self._device_pat.match(arg):
            return arg.replace(self.fake_type, self.real_type, 1)
        return arg

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        new_args = tuple(self._remap(a) for a in args)
        new_kwargs = {k: self._remap(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
```

### Modified files

#### `tests/conftest.py` — add `--emulate-xpu` plugin (Option 2)

Changes to the **root** conftest (not `tests/test_offload/conftest.py`):

```python
# --- NEW: added to tests/conftest.py ---
from types import SimpleNamespace

def pytest_addoption(parser):
    parser.addoption(
        "--emulate-xpu",
        action="store_true",
        default=False,
        help="Emulate XPU device identity on CUDA hardware via TorchFunctionMode",
    )


def pytest_configure(config):
    """Activate device emulation before test collection (before module imports).

    Three layers of patching:
      1. DeviceRemapMode — intercepts torch.* functions, remaps "xpu" → "cuda"
      2. Accelerator mock — torch.accelerator.current_accelerator() reports "xpu"
      3. is_accelerator_type patch — accepts both "xpu" and "cuda"

    Layer 3 is necessary because DeviceRemapMode converts torch.device("xpu") →
    torch.device("cuda"), so tensor.device.type is "cuda". But is_accelerator_type
    compares against the mocked "xpu" and would return False. This is the known
    trade-off: routing logic is bypassed and must be tested separately (Option 1).
    """
    if not config.getoption("--emulate-xpu"):
        return

    from tests.emulate_device import DeviceRemapMode

    real_type = torch.accelerator.current_accelerator().type  # "cuda"
    fake_type = "xpu"

    # Save originals for cleanup
    config._emulate_orig_current_accelerator = torch.accelerator.current_accelerator

    # Layer 1: DeviceRemapMode
    mode = DeviceRemapMode(fake_type=fake_type, real_type=real_type)
    mode.__enter__()
    config._emulate_device_remap_mode = mode

    # Layer 2: Mock accelerator identity
    fake_accel = SimpleNamespace(type=fake_type)
    torch.accelerator.current_accelerator = lambda: fake_accel

    # Layer 3: Patch is_accelerator_type to accept both types
    import compressed_tensors.offload.convert.helpers as helpers_mod
    config._emulate_orig_is_accelerator_type = helpers_mod.is_accelerator_type

    def patched_is_accelerator_type(device_type: str) -> bool:
        return device_type in (fake_type, real_type)

    helpers_mod.is_accelerator_type = patched_is_accelerator_type


def pytest_unconfigure(config):
    """Tear down device emulation — restore all patched objects."""
    mode = getattr(config, "_emulate_device_remap_mode", None)
    if mode is not None:
        mode.__exit__(None, None, None)

    orig_accel = getattr(config, "_emulate_orig_current_accelerator", None)
    if orig_accel is not None:
        torch.accelerator.current_accelerator = orig_accel

    orig_is_accel = getattr(config, "_emulate_orig_is_accelerator_type", None)
    if orig_is_accel is not None:
        import compressed_tensors.offload.convert.helpers as helpers_mod
        helpers_mod.is_accelerator_type = orig_is_accel
```

**No changes needed** to any existing test file or to `tests/test_offload/conftest.py`.
The emulation is fully opt-in and transparent.

#### Impact on existing `tests/conftest.py`

The current root `conftest.py` contains only `mock_per_group_calibration`,
`mock_per_channel_calibration`, and `mock_per_tensor_calibration` fixtures. The new
`pytest_addoption` / `pytest_configure` / `pytest_unconfigure` hooks are additive and
do not conflict.

### Files NOT changed

| File | Why unchanged |
|---|---|
| `tests/test_offload/conftest.py` | `assert_device_equal` compares `.type` strings — both sides see `"cuda"` under remap, so assertions pass without modification |
| `tests/testing_utils.py` | `requires_gpu` reads `torch.accelerator.device_count()` which is NOT mocked (real hardware count) — tests still skip correctly on CPU-only machines |
| All 18 module-level constant files | Constants evaluated after `pytest_configure` activates the mock + remap — they get `"xpu"` from the mock and `torch.device("cuda")` from the remap, exactly as designed |
| `src/` production code | Zero production changes needed |

### How `assert_device_equal` works under emulation

```python
# tests/test_offload/conftest.py:17-29
def assert_device_equal(device_a, device_b):
    ...
    assert device_a.type == device_b.type and a_index == b_index
```

Under `--emulate-xpu`:
- `tensor.device` → `cuda:0` (C-level, not intercepted)
- `ONLOAD_DEVICE` → `torch.device("xpu")` intercepted at import → `cuda:0`
- Both sides are `"cuda"` → assertion passes ✓

### How module-level constants resolve

```python
# Example: tests/test_offload/cache/test_device.py
_ACCEL_TYPE = torch.accelerator.current_accelerator().type
# → "xpu" (Layer 2 mock active before collection)

ONLOAD_DEVICE = torch.device(_ACCEL_TYPE)
# → torch.device("xpu") intercepted by Layer 1 → torch.device("cuda", None)

_DEV0 = f"{_ACCEL_TYPE}:0"
# → "xpu:0" (string, not a torch call — NOT intercepted at this point)
# BUT when used: tensor.to(_DEV0) → _remap("xpu:0") → "cuda:0" ✓
```

---

## Implementation Plan

1. **Phase 1** (PR 1): Option 1 — `test_xpu_routing.py` with mock routing tests (~80 lines)
2. **Phase 2** (PR 2): Option 2 — `emulate_device.py` + root conftest plugin with `--emulate-xpu` (~100 lines)
3. **Phase 3** (separate): Real XPU CI via Buildkite when ready

Phase 1 can ship immediately. Phase 2 requires CUDA hardware to validate.
