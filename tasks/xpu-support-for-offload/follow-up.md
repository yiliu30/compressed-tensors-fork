Hi @yiliu30 , thanks for preparing this. 
- This makes sense to me and sounds pretty straightforward for a refactor. I don't see requiring torch>=2.6 to be a big issue. Few questions:

- You maintain a table of the old/new APIs in the vllm RFC -- [RFC]: Replace torch.cuda API with torch.accelerator for better hardware compatiblity. vllm#30679. Some of this information is in your table above. Is that an exhaustive list of all usage across compressed-tensors and llm-compressor?
- "Once XPU CI is set up, we will switch to real XPU tests." <- This is the work we already have underway in a separate thread right?
- You don't list AMD, though given it plugs into torch.cuda backend we will still maintain tier 1 support for it right?
- What is your plan with vllm for the parts of the torch.cuda API which you have listed as NA in the unified torch.accelerator API? Is that ongoing work to reach feature parity? If we find CUDA-specific helper methods, for example sharing cuda memory pointers directly -- [Offload] Investigate using torch multiprocessing module for distributed offloading #656. What should our process be to make sure we retain feature parity across devices? We'd have to handle it case-by-base, or decide we will not use anything outside of the unified API.

>  requiring torch>=2.6 
Let me correct the torch requirements. As haihao mentioned, we need torch>=2.9. I saw current vllm using torch==2.10.
Would that not be a big issue?



3. **Minimum effective torch version for full migration:** torch 2.11 (for `torch.Stream`). If llm-compressor can target torch 2.10 for now, the Stream can use a conditional import or stay as `torch.cuda.Stream` with a backend check.
4. **Test migration is substantial in both repos** -- compressed-tensors has ~50+ `"cuda"` literal sites across 21 test files; llm-compressor has ~60+ across ~15 test files.



> XPU CI is set up
Yes, previous we have some plan to add XPU CI support in llm-compressor, we will extend the scope to include compressed-tensors as well.

> AMD support
Yes, it is plugs into torch.cuda backend, there should not a big gap here.


> 
exhaustive list of torch.cuda usages across compressed-tensors and llm-compressor
llm-compressor: /home/yiliu7/workspace/llm-compressor
compressed-tensors: /home/yiliu7/workspace/compressed-tensors
Key takeawys:

1. **compressed-tensors is clean** -- all 13 sites have direct replacements, no NA APIs. This is the straightforward migration in the current RFC.
2. **llm-compressor has 3 harder sites** -- `torch.cuda.Stream()` (needs torch 2.11), `torch.cuda.stream()` context (needs torch 2.10) for prefetching data and `torch.cuda.OutOfMemoryError` (no unified equivalent) for hanlding OoM.


- What is your plan with vllm for the parts of the torch.cuda API which you have listed as NA in the unified torch.accelerator API? Is that ongoing work to reach feature parity? If we find CUDA-specific helper methods, for example sharing cuda memory pointers directly -- [Offload] Investigate using torch multiprocessing module for distributed offloading #656. What should our process be to make sure we retain feature parity across devices? We'd have to handle it case-by-base, or decide we will not use anything outside of the unified API.


On the vLLM side, we're addressing the torch.accelerator / torch.cuda gap case-by-case. Most call sites have direct unified replacements, and for the   rest, vLLM's platform abstraction layer already provides the device-agnostic
   dispatch we need. We expect this gap to keep narrowing as PyTorch evolves —
   and where it doesn't, the exceptions should be few and well-contained. If
  the volume of edge cases grows, we can introduce a thin abstraction layer
  (similar to vLLM's current_platform) to keep things clean.


In vllm side, regarding the gap between torch.accelerator and torch.cuda, we are addressing it on a case-by-case basis. In most scenarios, we can rely on vLLM’s platform abstraction layer, which is designed to be device-agnostic. As Haihao mentioned, we expect this gap to narrow over time as PyTorch continues to evolve its unified APIs.


That said, using features outside the unified API is likely unavoidable in some cases, given the uneven pace of support across hardware vendors. However, we expect such exceptions to be limited and manageable.
ur suggestion is we can handle these gap case-by-case,As the number of cases increases.
we may introduce a think abstraction layer like the platform in vLLM.

In compressor-tensor side, we prepre start with replace all supportted `torch.accelerator` unified APIs. 
In llmc side, we'd like suggest three harder sites case by case. 




---

## Proposal: Handling `torch.cuda` APIs with No `torch.accelerator` Equivalent

### Context

The reviewer asks what our process should be when we encounter CUDA-specific APIs that have no unified `torch.accelerator` replacement today. The full inventory (see `torch-cuda-inventory.md`) identifies the concrete gaps:

| NA API | Repo | File | Impact |
|--------|------|------|--------|
| `torch.cuda.OutOfMemoryError` | llm-compressor | `pipelines/sequential/helpers.py` | OOM error handling |
| `torch.cuda.Stream()` | llm-compressor | `pipelines/cache.py` | H2D prefetch streams |
| `torch.cuda.stream()` context | llm-compressor | `pipelines/cache.py` | Stream context manager |
| `torch.cuda.CUDAGraph` | vLLM (not in ct/llm-c) | — | Graph capture |
| CUDA memory pointer sharing | vLLM #656 (future) | — | Distributed offloading |

**Key observation: compressed-tensors has zero NA gaps.** All 13 CUDA-specific sites have direct `torch.accelerator` replacements. The NA problem is confined to llm-compressor and vLLM.

### Proposed approach: Case-by-case with a tiered policy

We propose handling NA APIs through a three-tier policy, applied per-API based on the nature of the gap:

#### Tier 1: Use `torch.accelerator` directly (no gap)

For APIs where `torch.accelerator` provides a direct replacement, use it unconditionally. This covers all of compressed-tensors and the majority of llm-compressor.

Examples: `is_available()`, `device_count()`, `current_device()`, `set_device()`, `get_memory_info()`, `max_memory_allocated()`, `reset_peak_memory_stats()`, `Event()`, `current_stream()`.

**Action:** Direct replacement. No abstraction needed.

#### Tier 2: Use backend-specific API behind a thin helper (gap is closing)

For APIs where PyTorch is actively working on a unified replacement but it hasn't shipped yet in our minimum supported version, wrap the call in a small helper function. When the unified API becomes available at our minimum torch version, swap the implementation — one-line change, no call-site updates.

| API | Unified replacement | Available since | Status |
|-----|-------------------|-----------------|--------|
| `torch.cuda.Stream()` | `torch.Stream(device_type)` | torch 2.11 | Shipped |
| `torch.cuda.stream()` | `with stream:` context manager | torch 2.10 | Shipped |
| `torch.cuda.OutOfMemoryError` | `torch.OutOfMemoryError` | torch 2.5 | **Already shipped** |

Concrete example for `OutOfMemoryError` — this is already solved:

```python
# llm-compressor: pipelines/sequential/helpers.py
# Before:
except torch.cuda.OutOfMemoryError as e:
    raise torch.cuda.OutOfMemoryError(...)

# After (works today, torch >= 2.5):
except torch.OutOfMemoryError as e:
    raise torch.OutOfMemoryError(...)
```

`torch.OutOfMemoryError` is the device-agnostic base class. `torch.cuda.OutOfMemoryError` and `torch.xpu.OutOfMemoryError` both inherit from it. Catching the base class handles all backends.

For `Stream` and `stream()` context — if our minimum torch is 2.10+, these are also already solved:

```python
# Before:
h2d_stream = torch.cuda.Stream()
with torch.cuda.stream(h2d_stream):
    ...

# After (torch >= 2.10):
h2d_stream = torch.Stream(device_type)
with h2d_stream:
    ...
```

**Action:** Replace directly if minimum torch version covers it. If not, wrap in a helper that we swap later.

#### Tier 3: Delegate to vLLM's platform abstraction (inherently device-specific)

For APIs that are fundamentally device-specific and unlikely to ever have a unified PyTorch equivalent, defer to vLLM's `current_platform` abstraction. This is the right layer for these because:

- vLLM already maintains per-backend platform implementations (CUDA, ROCm, XPU, TPU, etc.)
- Each platform registers its own capabilities and can provide backend-specific optimizations
- compressed-tensors and llm-compressor don't need to duplicate this

Examples:
- **CUDAGraph / graph capture** — XPU has its own graph API, TPU has its own compilation model. vLLM's platform layer already abstracts this.
- **CUDA memory pointer sharing** (#656) — this is a CUDA IPC feature. XPU has `xe_ipc`, NPU has its own mechanism. If compressed-tensors needs this for distributed offloading, it should go through vLLM's platform layer rather than introducing a new abstraction.
- **Device-specific profiling** (`torch.cuda.nvtx`, `torch.cuda.profiler`) — each backend has its own profiling tools.

**Action:** These features are not used in compressed-tensors or llm-compressor production code today. If they are introduced in the future (e.g., #656), route through vLLM's platform abstraction.

### Decision process for new CUDA-specific code

When a contributor introduces a new `torch.cuda.*` call, the review checklist is:

1. **Check `torch.accelerator`** — does a unified equivalent exist at our minimum torch version? If yes, use it. (Tier 1)
2. **Check PyTorch roadmap** — is a unified equivalent shipped in a newer torch version? If yes, wrap in a helper with a TODO to unwrap when we bump minimum torch. (Tier 2)
3. **Is it inherently device-specific?** If yes, it belongs in vLLM's platform layer, not in compressed-tensors or llm-compressor. (Tier 3)
4. **If none of the above** — use the backend-specific API gated behind `if torch.accelerator.current_accelerator().type == "cuda":`, with a clear comment explaining why and a tracking issue for the gap.

### What this means for each repo

| Repo | NA gaps | Resolution |
|------|---------|------------|
| **compressed-tensors** | None | Fully migrated to `torch.accelerator`. No exceptions needed. |
| **llm-compressor** | 3 sites | `OutOfMemoryError` → use `torch.OutOfMemoryError` (Tier 2, already available). `Stream`/`stream()` → use `torch.Stream` + `with stream:` (Tier 2, available at torch 2.10+). |
| **vLLM** | CUDAGraph, memory pointers, profiling | Handled by vLLM's existing platform abstraction (Tier 3). Separate migration tracked in vllm#30679. |

### Summary

The gap between `torch.accelerator` and `torch.cuda` is real but narrowing fast. For compressed-tensors, there is no gap at all. For llm-compressor, the remaining 3 NA sites all have solutions available today (torch 2.5+ for OOM, torch 2.10+ for Stream). For inherently device-specific features, vLLM's platform abstraction is the right place to handle them.

We do not need a new abstraction layer in compressed-tensors or llm-compressor. The combination of `torch.accelerator` (Tier 1 + 2) and vLLM's platform layer (Tier 3) covers all current and foreseeable cases.
