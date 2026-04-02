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

        # Activate DeviceRemapMode BEFORE test module imports
        # This intercepts torch.device() in module-level constants
        mode = DeviceRemapMode(fake_type=fake_type, real_type=real_type)
        mode.__enter__()
        config._device_remap_mode = mode

        # Mock accelerator to report fake identity
        fake_accel = SimpleNamespace(type=fake_type)
        torch.accelerator.current_accelerator = lambda: fake_accel
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

#### Verified end-to-end (on XPU hardware, emulating CUDA identity)

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

## Implementation Plan

1. **Phase 1** (PR 1): Option 1 — `test_xpu_routing.py` with mock routing tests (~60 lines)
2. **Phase 2** (PR 2): Option 2 — `DeviceRemapMode` conftest plugin with `--emulate-xpu` flag (~80 lines)
3. **Phase 3** (separate): Real XPU CI via Buildkite when ready

Phase 1 can ship immediately. Phase 2 can be deferred if the team prefers simplicity.
