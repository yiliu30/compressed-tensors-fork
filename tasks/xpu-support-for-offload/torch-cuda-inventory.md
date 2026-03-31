
# Exhaustive `torch.cuda` Usage Inventory

Scanned from `origin/main` for both repos on 2026-03-31.

---

## compressed-tensors

### Production code (`src/compressed_tensors/`)

| File | Line | `torch.cuda` usage | `torch.accelerator` replacement | Status |
|------|------|-------------------|--------------------------------|--------|
| `offload/dispatch.py` | 240 | `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Direct |
| `offload/dispatch.py` | 248, 253 | `torch.cuda.get_device_properties(i).total_memory` | `torch.accelerator.get_memory_info(i)[1]` | Direct |
| `offload/dispatch.py` | 254 | `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Direct |
| `offload/dispatch.py` | 249 | `torch.device("cuda")` literal | `torch.device(accel_type, idx)` | String literal |
| `offload/load.py` | 93, 96 | `torch.cuda.get_device_properties(i).total_memory` | `torch.accelerator.get_memory_info(i)[1]` | Direct |
| `offload/load.py` | 97 | `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Direct |
| `offload/dist_utils.py` | 33 | `torch.cuda.set_device(device)` | `torch.accelerator.set_device_index(local_rank)` | Direct |
| `offload/dist_utils.py` | 35 | `backend="nccl"` | `dist.get_default_backend_for_device(device)` | Direct |
| `offload/cache/disk.py` | 179 | `device.type in ("cuda")` | `is_accelerator_type(device.type)` | String literal |
| `offload/cache/disk.py` | 181 | `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Direct |
| `offload/cache/base.py` | 74, 76 | `"cuda"` match arms | `is_accelerator_type()` capability check | String literal |
| `offload/convert/helpers.py` | 37 | `device.type == "cuda"` | `is_accelerator_type(device.type)` | String literal |

**Totals:** 8 `torch.cuda.*` API calls + 4 `"cuda"` string literals + 1 `"nccl"` literal = **13 CUDA-specific sites**.
All have direct `torch.accelerator` replacements. **No NA APIs used.**

### Test code (`tests/`)

#### `torch.cuda` API calls

| File | Line | Usage | Migration |
|------|------|-------|-----------|
| `testing_utils.py` | 109 | `torch.cuda.device_count() > 0` | `torch.accelerator.is_available()` |
| `testing_utils.py` | 136 | `torch.cuda.device_count() < num_required_gpus` | `torch.accelerator.device_count()` |
| `test_offload/conftest.py` | 26 | `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` |
| `test_offload/conftest.py` | 75 | `torch.cuda.set_device(local_rank)` | `torch.accelerator.set_device_index(local_rank)` |
| `test_offload/test_dispatch.py` | 215, 238 | `patch("...torch.cuda")` mock target | Retarget to `torch.accelerator` |
| `test_compressors/test_compress_decompress_module.py` | 18 | `torch.cuda.is_available()` | `torch.accelerator.is_available()` |

#### `"cuda"` string literals (need `accel_device` fixture)

| File | Lines |
|------|-------|
| `test_offload/cache/test_cpu.py` | 22 |
| `test_offload/cache/test_device.py` | 25, 30 |
| `test_offload/cache/test_disk.py` | 28 |
| `test_offload/cache/test_dist_cpu.py` | 26 |
| `test_offload/cache/test_dist_device.py` | 31, 36 |
| `test_offload/cache/test_dist_disk.py` | 30, 210 |
| `test_offload/test_dispatch.py` | 81, 86, 94, 95, 102, 103, 111, 112, 119, 128, 129, 138, 139, 140, 148, 168, 169, 170, 182, 186, 188, 193, 196, 203, 207 |
| `test_offload/test_interface.py` | 25 |
| `test_offload/test_load.py` | 28, 32, 34, 35 |
| `test_offload/test_module.py` | 17 |
| `test_offload/conftest.py` | 111, 113 |
| `test_offload/convert/test_to_accelerate.py` | 19, 25, 27, 30, 31, 46, 47, 48 |
| `test_offload/convert/test_from_accelerate.py` | 34, 52, 80 |
| `test_compressors/test_compress_decompress_module.py` | 18, 77, 93 |
| `test_modeling/test_attention_and_cache.py` | 23, 25 |
| `test_modeling/test_deepseekv3_kvcache_quant.py` | 28, 30 |
| `test_quantization/lifecycle/test_initialize.py` | 111 |
| `test_transform/factory/test_memory.py` | 34 |
| `test_transform/factory/test_serialization.py` | 26, 41, 67 |
| `test_transform/factory/test_correctness.py` | 90, 95, 186, 197, 212 |
| `test_transform/utils/test_hadamard.py` | 35, 37, 71, 75, 77 |

---

## llm-compressor

### Production code (`src/llmcompressor/`)

| File | Line | `torch.cuda` usage | `torch.accelerator` replacement | Complexity |
|------|------|-------------------|--------------------------------|------------|
| `entrypoints/model_free/helpers.py` | 26 | `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Direct |
| `entrypoints/model_free/helpers.py` | 27 | `torch.device("cuda:0")` literal | `torch.device(accel_type, 0)` | Direct |
| `datasets/utils.py` | 141 | `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Direct |
| `modifiers/autoround/base.py` | 357 | `torch.cuda.is_available()`, `f"cuda:{rank}"` | `torch.accelerator.is_available()`, `torch.device(accel_type, rank)` | Direct |
| `modifiers/awq/base.py` | 1045 | `torch.cuda.current_device()`, `f"cuda:{...}"` | `torch.accelerator.current_device_index()`, `torch.device(accel_type, ...)` | Direct |
| `pipelines/cache.py` | 223 | `torch.cuda.Stream()`, `torch.cuda.is_available()` | `torch.Stream(accel_type)` | **Needs torch 2.11** |
| `pipelines/cache.py` | 228 | `torch.cuda.stream(h2d_stream)` | `with stream:` context manager | **Needs torch 2.10+** |
| `pipelines/cache.py` | 230 | `torch.cuda.Event()` | `torch.Event(accel_type)` | Direct (torch 2.9+) |
| `pipelines/cache.py` | 250 | `torch.cuda.current_stream().wait_event(event)` | `torch.accelerator.current_stream().wait_event(event)` | Direct (torch 2.9+) |
| `pipelines/cache.py` | 277 | `torch.device(device).type == "cuda"` | `is_accelerator_type(...)` | String literal |
| `pipelines/cache.py` | 326 | `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Direct |
| `pipelines/sequential/helpers.py` | 553-554 | `torch.cuda.OutOfMemoryError` | **NA** -- no unified equivalent | **NA** |
| `utils/metric_logging.py` | 52 | `torch.cuda.max_memory_allocated(device_id)` | `torch.accelerator.max_memory_allocated(device_id)` | Direct (torch 2.9+) |
| `utils/metric_logging.py` | 53 | `torch.cuda.get_device_properties(device_id).total_memory` | `torch.accelerator.get_memory_info(device_id)[1]` | Direct |
| `utils/metric_logging.py` | 66 | `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Direct |
| `utils/metric_logging.py` | 69 | `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Direct |
| `utils/pytorch/utils.py` | 13 | `torch.cuda.reset_peak_memory_stats(self.device)` | `torch.accelerator.reset_peak_memory_stats(self.device)` | Direct (torch 2.9+) |
| `utils/pytorch/utils.py` | 18, 23 | `torch.cuda.max_memory_allocated(self.device)` | `torch.accelerator.max_memory_allocated(self.device)` | Direct (torch 2.9+) |
| `utils/dev.py` | 129-130 | `torch.cuda.is_available()`, `f"cuda:{rank}"` | `torch.accelerator.is_available()`, `torch.device(accel_type, rank)` | Direct |

**Totals:** 19 `torch.cuda.*` call sites across 8 files + 3 `"cuda"` string literals = **22 CUDA-specific sites**.

**Breakdown by migration difficulty:**
- **Direct replacement (17 sites):** `is_available`, `device_count`, `current_device`, `max_memory_allocated`, `reset_peak_memory_stats`, `get_device_properties().total_memory`, `Event`, `current_stream`
- **Needs torch 2.10+ (1 site):** `torch.cuda.stream()` context manager -> `with stream:`
- **Needs torch 2.11 (1 site):** `torch.cuda.Stream()` -> `torch.Stream(accel_type)`
- **NA -- no unified equivalent (1 site):** `torch.cuda.OutOfMemoryError` -- needs device-conditional handling
- **String literals (3 sites):** `"cuda"` / `f"cuda:{...}"` -- use `accel_type` variable

### Test code (`tests/`)

| File | Lines | Usage |
|------|-------|-------|
| `testing_utils.py` | 60 | `torch.cuda.device_count()` |
| `testing_utils.py` | 108-109 | `torch.cuda.mem_get_info()`, `torch.cuda.device_count()` |
| `lmeval/test_lmeval.py` | 109-110 | `torch.cuda.is_available()`, `torch.cuda.manual_seed_all()` |
| `e2e/vLLM/test_vllm.py` | 149-150 | `torch.cuda.empty_cache()`, `torch.cuda.synchronize()` |
| `llmcompressor/pipelines/test_cache.py` | 141 | `torch.cuda.is_available()` |
| `llmcompressor/pytorch/utils/test_helpers.py` | 174, 210 | `torch.cuda.is_available()` skip decorators |
| `llmcompressor/transformers/gptq/test_gptq_oneshot.py` | 67 | `torch.cuda.is_available()` |
| `llmcompressor/transformers/sparsegpt/test_sparsegpt_lm_head.py` | 13, 45 | `torch.cuda.is_available()` |
| `llmcompressor/transformers/sparsegpt/test_sparsegpt_sparsity.py` | 24 | `torch.cuda.empty_cache()` |
| `llmcompressor/transformers/compression/test_has_gpu.py` | 13 | `torch.cuda.is_available()` |
| `llmcompressor/transformers/autoround/test_autoround_oneshot.py` | 102, 148, 210 | `torch.cuda.is_available()` |
| `llmcompressor/transformers/compression/test_run_compressed.py` | 48-49 | `torch.cuda.empty_cache()`, `torch.cuda.synchronize()` |
| `llmcompressor/transformers/compression/test_quantization.py` | 93-94 | `torch.cuda.empty_cache()`, `torch.cuda.synchronize()` |
| `llmcompressor/modifiers/quantization/test_quantization_ddp.py` | 19, 23 | `torch.cuda.is_available()`, `torch.cuda.set_device()` |

**Plus ~60+ `"cuda"` string literal sites** across test files (device_map, torch.device, YAML configs).

---

## Cross-repo summary

| `torch.cuda` API | compressed-tensors (prod) | llm-compressor (prod) | `torch.accelerator` replacement | Since |
|---|:---:|:---:|---|---|
| `is_available()` | 1 | 5 | `torch.accelerator.is_available()` | 2.6 |
| `device_count()` | 2 | 1 | `torch.accelerator.device_count()` | 2.6 |
| `current_device()` | 1 | 2 | `torch.accelerator.current_device_index()` | 2.6 |
| `set_device()` | 1 | 0 | `torch.accelerator.set_device_index()` | 2.6 |
| `get_device_properties().total_memory` | 3 | 1 | `torch.accelerator.get_memory_info()[1]` | 2.9 |
| `max_memory_allocated()` | 0 | 3 | `torch.accelerator.max_memory_allocated()` | 2.9 |
| `reset_peak_memory_stats()` | 0 | 1 | `torch.accelerator.reset_peak_memory_stats()` | 2.9 |
| `Stream()` | 0 | 1 | `torch.Stream(accel_type)` | **2.11** |
| `stream()` context | 0 | 1 | `with stream:` | 2.10 |
| `Event()` | 0 | 1 | `torch.Event(accel_type)` | 2.9 |
| `current_stream()` | 0 | 1 | `torch.accelerator.current_stream()` | 2.9 |
| `OutOfMemoryError` | 0 | 1 | **NA** | NA |
| `backend="nccl"` | 1 | 0 | `dist.get_default_backend_for_device()` | 2.6 |
| `"cuda"` string literals | 4 | 3 | `accel_type` / `is_accelerator_type()` | -- |
| **Total** | **13** | **22** | | |

### Key takeaways

1. **compressed-tensors is clean** -- all 13 sites have direct replacements, no NA APIs. This is the straightforward migration in the current RFC.
2. **llm-compressor has 3 harder sites** -- `torch.cuda.Stream()` (needs torch 2.11), `torch.cuda.stream()` context (needs torch 2.10), and `torch.cuda.OutOfMemoryError` (no unified equivalent). These are all in `pipelines/cache.py` and `pipelines/sequential/helpers.py`.
3. **Minimum effective torch version for full migration:** torch 2.11 (for `torch.Stream`). If llm-compressor can target torch 2.10 for now, the Stream can use a conditional import or stay as `torch.cuda.Stream` with a backend check.
4. **Test migration is substantial in both repos** -- compressed-tensors has ~50+ `"cuda"` literal sites across 21 test files; llm-compressor has ~60+ across ~15 test files.

---

## Concise API mapping

### compressed-tensors

| `torch.cuda` | Replacement | Since |
|---|---|---|
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | 2.6 |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | 2.6 |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | 2.6 |
| `torch.cuda.set_device(dev)` | `torch.accelerator.set_device_index(idx)` | 2.6 |
| `torch.cuda.get_device_properties(i).total_memory` | `torch.accelerator.get_memory_info(i)[1]` | 2.9 |
| `backend="nccl"` | `dist.get_default_backend_for_device(device)` | 2.6 |
| `device.type == "cuda"` / `"cuda"` literals | `is_accelerator_type(device.type)` | -- |
| `torch.device(f"cuda:{idx}")` | `torch.device(accel_type, idx)` | -- |

**All 13 sites have direct replacements. No NA gaps.**

### llm-compressor

| `torch.cuda` | Replacement | Since | Notes |
|---|---|---|---|
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | 2.6 | |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | 2.6 | |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | 2.6 | |
| `torch.cuda.get_device_properties(i).total_memory` | `torch.accelerator.get_memory_info(i)[1]` | 2.9 | |
| `torch.cuda.max_memory_allocated(dev)` | `torch.accelerator.max_memory_allocated(dev)` | 2.9 | |
| `torch.cuda.reset_peak_memory_stats(dev)` | `torch.accelerator.reset_peak_memory_stats(dev)` | 2.9 | |
| `torch.cuda.Event()` | `torch.Event(accel_type)` | 2.9 | |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | 2.9 | |
| `torch.cuda.stream(s)` | `with stream:` context manager | 2.10 | |
| `torch.cuda.Stream()` | `torch.Stream(accel_type)` | **2.11** | Needs newer torch |
| `torch.cuda.OutOfMemoryError` | **NA** | -- | XPU has no `torch.xpu.OutOfMemoryError`; `torch.OutOfMemoryError` exists but XPU does not raise it (verified on 2.10.0+xpu). Needs `RuntimeError` message matching as fallback. |
| `device.type == "cuda"` / `"cuda"` literals | `is_accelerator_type(device.type)` | -- | |
| `torch.device(f"cuda:{idx}")` | `torch.device(accel_type, idx)` | -- | |

**19 of 22 sites have direct replacements. 3 sites need special handling (Stream, stream context, OutOfMemoryError).**
