# RFC: Multi-Accelerator Offload Support via `torch.accelerator`

## Summary

Replace all hardcoded `torch.cuda.*` calls in `compressed_tensors.offload` with
`torch.accelerator` (PyTorch 2.6+). This enables XPU, Ascend NPU, Gaudi HPU, and
any future backend — with **zero new abstraction code** and no new files.

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
| `backend="nccl"` | `dist.Backend.default_device_backend_map[accel_type]` |

The distributed backend map is maintained by PyTorch: `cuda→nccl`, `xpu→xccl`,
`npu→hccl`, `mps→gloo`.

## Scope

The offload system is 90% device-agnostic already. All cache implementations
(`CPUCache`, `DeviceCache`, `DiskCache`, and distributed variants) work for any
device via `tensor.to()`. Changes are confined to **6 production files, ~35 lines**.

## Production Changes

### 1. `cache/base.py` — `cls_from_device()`

Replace explicit `"cuda"` match arms with exclusion-based guard:

```python
def is_accelerator_type(device_type: str) -> bool:
    return device_type not in ("cpu", "meta", "disk")

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
backend = dist.Backend.default_device_backend_map[accel_type]
dist.init_process_group(backend=backend, ...)
```

### 6. `convert/helpers.py` — `norm_device()`

Replace `device.type == "cuda"` with `is_accelerator_type(device.type)`.

## Test Changes (~40 lines across 14 files)

Mechanical: replace `torch.device("cuda")` → `accel_device` fixture backed by
`torch.accelerator.current_accelerator()`. Rename `cuda_device` → `accel_device`.
Update `requires_gpu` to use `torch.accelerator.is_available()`.

## Risks

- **PyTorch version:** `torch.accelerator` requires ≥2.6. Current minimum is 1.7.0.
  Recommend bumping — any XPU/NPU user is already on ≥2.6.
- **safetensors:** Verified 0.7.0 accepts `"xpu"` and `"xpu:0"`. NPU/HPU string
  support unverified — fallback is load-to-CPU then `.to(device)`.
- **Distributed:** `xccl`/`hccl` maturity varies. `dist.broadcast`/`dist.barrier`
  need per-backend validation.
