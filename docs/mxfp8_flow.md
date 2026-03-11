# MXFP8 Quantization Flow: Observer → Disk

This document traces how weights flow through the MXFP8 quantization pipeline,
from observer calibration to compressed safetensors on disk.

## Overview

```
oneshot(model, recipe="MXFP8")
  │
  ├─ 1. Apply quantization config to model (attach schemes, observers, scales/zp buffers)
  ├─ 2. Calibrate: observer computes scales & zero_points from weight statistics
  ├─ 3. Forward pass: fake_quantize simulates quantization loss (quantize → dequantize)
  ├─ 4. Freeze: delete observers, lock scales
  │
  └─ model.save_pretrained(save_dir, save_compressed=True)
       ├─ 5. Compress: float weight → fp8 quantized weight, float scale → uint8 E8M0
       ├─ 6. Save: write safetensors + config.json
       └─ 7. Update config: write quantization_config to config.json
```

---

## Phase 1: Initialization

**Entry:** `apply_quantization_config(model, config)`
→ `initialize_module_for_quantization(module)`

For each `Linear` layer (except `lm_head`), the system:

1. Attaches a `QuantizationScheme` to the module:
   ```python
   module.quantization_scheme = QuantizationScheme(
       weights=QuantizationArgs(
           num_bits=8, type="float", strategy="group",
           group_size=32, scale_dtype=torch.uint8, ...
       ),
       input_activations=QuantizationArgs(
           num_bits=8, type="float", strategy="group",
           group_size=32, dynamic=True, ...
       ),
   )
   ```

2. Registers scale/zero_point buffers on the module:
   ```python
   # For a Linear with shape [out_features, in_features]:
   module.weight_scale     # shape: [out_features, in_features // 32]
   module.weight_zero_point  # shape: [out_features, in_features // 32]
   ```

3. Attaches an observer (e.g., `memoryless_minmax`) as a submodule:
   ```python
   module.weight_observer = Observer(base_name="weight", args=weight_quant_args)
   ```

4. Sets `module.quantization_status = QuantizationStatus.INITIALIZED`

**Source:** `compressed_tensors/quantization/lifecycle/initialize.py`

---

## Phase 2: Calibration (Observer Computes Scales)

**Entry:** `update_weight_zp_scale(module)` → `call_observer(module, "weight")`

### 2a. Observer finds min/max per group

```python
# observer.forward(weight) internally calls:
observed = weight.reshape(*weight.shape[:-1], -1, group_size)  # reshape into groups of 32
min_vals = torch.amin(observed, dim=-1)  # per-group min
max_vals = torch.amax(observed, dim=-1)  # per-group max
```

### 2b. `calculate_qparams()` computes scale from min/max

```python
# compressed_tensors/quantization/utils/helpers.py

min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
max_vals = torch.max(max_vals, torch.zeros_like(max_vals))
max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))

# For MXFP8: should_generate_mx_scales() returns True
#   (because type="float", group_size=32, scale_dtype=uint8)

scales = generate_mx_scales(x=max_val_pos, num_bits=8)
```

### 2c. `generate_mx_scales()` produces E8M0 exponents

```python
# compressed_tensors/quantization/utils/mxfp4_utils.py

def generate_mx_scales(x, num_bits=8):
    offset = _MX_ELEM_OFFSET[8]  # = 8, because floor(log2(FP8_max=448)) = 8
    scale_power_2 = round_to_power_2(x)   # snap to nearest power of 2
    return 127 + torch.floor(torch.log2(scale_power_2)) - offset
    #      ^^^                                            ^^^^^
    #   E8M0 bias                                    shift so group max
    #                                                maps into FP8 range
```

**Example:** group max = 5.0
```
round_to_power_2(5.0) = 4.0
log2(4.0) = 2.0
E8M0 exponent = 127 + 2 - 8 = 121  (stored as uint8)
actual scale = 2^(121-127) = 2^(-6) = 0.015625
5.0 / 0.015625 = 320.0  (fits in FP8 range [-448, 448]) ✓
```

### 2d. Round to uint8, then convert back to float

```python
# Still in calculate_qparams():

# Round to scale_dtype (uint8)
scales = round_to_quantized_type_dtype(scales, dtype=torch.uint8)
# scales is now: uint8 E8M0 exponents (e.g., tensor([121], dtype=uint8))

# Convert back to float for use in QDQ
scales = maybe_convert_from_mx_exp(quantization_args, scales)
# scales is now: float power-of-2 values (e.g., tensor([0.015625]))
```

### 2e. Store on module

```python
# calibration.py: call_observer()
scale, zero_point = observer(value)
update_offload_parameter(module, "weight_scale", scale)       # float power-of-2
update_offload_parameter(module, "weight_zero_point", zero_point)  # zeros (symmetric)
```

**At this point:** `module.weight_scale` holds **float** power-of-2 values (not uint8).

**Source:** `llmcompressor/modifiers/quantization/calibration.py`

---

## Phase 3: Forward Pass (Fake Quantize)

**Entry:** `quantized_forward()` → `forward_quantize()` → `fake_quantize()`

During calibration forward passes, the weight is fake-quantized (quantize then
immediately dequantize) to simulate quantization error:

```python
# forward.py: forward_quantize()
scale = module.weight_scale          # float power-of-2 from observer
zero_point = module.weight_zero_point  # zeros

fake_quantize(x=weight, scale=scale, zero_point=zero_point, args=quant_args)
```

### `fake_quantize` = `_quantize` + `_dequantize`:

```python
# For GROUP strategy with group_size=32:
x = weight.unflatten(-1, (num_groups, 32))        # reshape into groups
scale = scale.unsqueeze(-1)                        # broadcast over group dim

# _quantize:
scaled = x / scale                                  # divide by scale
quantized = round_to_quantized_type(scaled, ...)    # round + clamp to FP8 range [-448, 448]
quantized = quantized.to(torch.float8_e4m3fn)       # cast to FP8 dtype

# _dequantize:
dequantized = quantized.to(scale.dtype) * scale     # multiply back by scale

output = dequantized.flatten(start_dim=-2)           # reshape back
```

**Math:**
```
x_q = clamp(round(x / scale), -448, 448)   →  quantized FP8 value
x̂   = x_q × scale                          →  dequantized (with quantization error)
```

The model trains/calibrates with `x̂` so downstream layers see realistic quantization noise.

**Source:** `compressed_tensors/quantization/lifecycle/forward.py`

---

## Phase 4: Freeze

**Entry:** `freeze_module_quantization(module)`

```python
# Delete all observers (no longer needed)
del module.weight_observer
del module.input_observer
# etc.

module.quantization_status = QuantizationStatus.FROZEN
```

After freezing, `module.weight_scale` (float) and `module.weight_zero_point` are
permanent parameters on the module.

**Source:** `llmcompressor/modifiers/quantization/calibration.py`

---

## Phase 5: Compression (Float → Disk Format)

**Entry:** `model.save_pretrained(save_dir, save_compressed=True)`

This calls a patched `save_pretrained_wrapper` (from `modify_save_pretrained`):

```python
# llmcompressor/transformers/compression/compressed_tensors_utils.py

compressor = get_model_compressor(model, ...)
#   → ModelCompressor.from_pretrained_model(model)
#     → infers format "mxfp8-quantized" from QuantizationScheme
#     → creates MXFP8QuantizationCompressor

compressor.compress_model(model)
#   → ModelCompressor.compress(model)
```

### 5a. `ModelCompressor.compress()` gets state_dict and delegates

```python
# model_compressor.py
state_dict = model.state_dict()
# state_dict contains:
#   "model.layers.0.mlp.gate_proj.weight"       → bfloat16 [4864, 896]
#   "model.layers.0.mlp.gate_proj.weight_scale"  → float power-of-2 [4864, 28]
#   "model.layers.0.mlp.gate_proj.weight_zero_point" → zeros [4864, 28]
#   ...

state_dict = quant_compressor.compress(state_dict, names_to_scheme=module_to_scheme)
```

### 5b. `BaseQuantizationCompressor.compress()` iterates over layers

```python
# base.py
for name in state_dict:
    if name.endswith("weight"):
        scale = state_dict[prefix + "weight_scale"]      # float power-of-2
        zp = state_dict[prefix + "weight_zero_point"]
        compressed_values = self.compress_weight(
            weight=value, scale=scale, zero_point=zp,
            quantization_args=quant_args,
        )
        # compressed_values = {"weight": fp8_tensor, "weight_scale": uint8_tensor}
        for key, value in compressed_values.items():
            compressed_dict[prefix + key] = value

    elif name.endswith("weight_scale") and self._skip_scale():
        continue  # MXFP8 compressor handles scale in compress_weight, skip duplicate
```

**Key:** `_skip_scale()` returns `True` for `MXFP8QuantizationCompressor`, so the
original float `weight_scale` from the state_dict is NOT written to the compressed
dict. Instead, the uint8 E8M0 scale from `compress_weight()` replaces it.

### 5c. `MXFP8QuantizationCompressor.compress_weight()` does the actual compression

```python
# naive_quantized.py

class MXFP8QuantizationCompressor(NaiveQuantizationCompressor):

    def compress_weight(self, weight, scale, quantization_args, ...):
        # Step 1: Parent quantizes weight to FP8 using float scale
        result = super().compress_weight(weight, scale, ...)
        # result["weight"] is now float8_e4m3fn
        #   internally: quantize(x=weight, scale=scale, ..., dtype=float8_e4m3fn)
        #     → clamp(round(weight / scale), -448, 448).to(float8_e4m3fn)

        # Step 2: Convert float scale → E8M0 uint8 for storage
        scale_exp = 127 + torch.floor(torch.log2(scale)).to(torch.int32)
        result["weight_scale"] = scale_exp.to(torch.uint8)

        return result
        # Returns:
        #   "weight"       → float8_e4m3fn [out_features, in_features]
        #   "weight_scale" → uint8 E8M0    [out_features, in_features // 32]
```

**Example:**
```
float scale = 0.015625 = 2^(-6)
E8M0 = 127 + floor(log2(0.015625)) = 127 + (-6) = 121
stored as uint8: 121
```

**Source:** `compressed_tensors/compressors/quantized_compressors/naive_quantized.py`

---

## Phase 6: Save to Disk

### 6a. Write safetensors

```python
# Back in save_pretrained_wrapper:
original_save_pretrained(save_directory, ...)
# → transformers writes compressed_dict to model.safetensors
```

**What's on disk (model.safetensors):**
```
model.layers.0.mlp.gate_proj.weight       → float8_e4m3fn [4864, 896]
model.layers.0.mlp.gate_proj.weight_scale  → uint8         [4864, 28]
model.layers.0.mlp.gate_proj.bias         → bfloat16      [4864]  (if exists)
...
(weight_zero_point is NOT saved — symmetric quantization, all zeros)
```

### 6b. Write config.json

```python
# save_pretrained_wrapper:
compressor.update_config(save_directory)
# → writes quantization_config into config.json
```

**What's in config.json:**
```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "format": "mxfp8-quantized",
    "config_groups": {
      "group_0": {
        "format": "mxfp8-quantized",
        "targets": ["Linear"],
        "weights": {
          "num_bits": 8,
          "type": "float",
          "strategy": "group",
          "group_size": 32,
          "symmetric": true,
          "scale_dtype": "torch.uint8"
        },
        "input_activations": {
          "num_bits": 8,
          "type": "float",
          "strategy": "group",
          "group_size": 32,
          "dynamic": true,
          "scale_dtype": "torch.uint8"
        }
      }
    },
    "ignore": ["lm_head"]
  }
}
```

---

## Phase 7: Loading Back (Decompression)

When loading a saved MXFP8 model, the reverse happens:

```python
model = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-MXFP8")
```

### 7a. Transformers reads config.json, sees `quant_method: "compressed-tensors"`

→ Creates `CompressedTensorsHfQuantizer` → Creates `ModelCompressor`
→ `apply_quantization_config(model)` attaches schemes
→ Loads safetensors weights into model (fp8 weight + uint8 scale)

### 7b. `MXFP8QuantizationCompressor.decompress_weight()` restores float weights

```python
def decompress_weight(self, compressed_data, quantization_args=None):
    scale = compressed_data["weight_scale"]       # uint8 E8M0

    # Convert E8M0 → float
    scale_exp = scale.to(torch.int32) - 127       # biased → unbiased exponent
    scale_float = 2.0 ** scale_exp.to(torch.float)  # → power-of-2 float

    compressed_data["weight_scale"] = scale_float

    # Parent dequantizes: weight_float = fp8_weight.to(float) * scale_float
    return super().decompress_weight(compressed_data, quantization_args)
```

**Example:**
```
uint8 scale = 121
exponent = 121 - 127 = -6
float scale = 2^(-6) = 0.015625
fp8 weight value = 1.5 (in float8_e4m3fn)
dequantized = 1.5 × 0.015625 = 0.0234375
```

---

## Summary: Data Type Transformations

```
                        CALIBRATION
weight (bf16) ─────────────────────────────────→ weight (bf16, unchanged)
                  │
                  ├─ observer: min/max per group
                  ├─ generate_mx_scales() → E8M0 uint8 exponents
                  ├─ round_to_quantized_type() → uint8
                  ├─ maybe_convert_from_mx_exp() → float power-of-2
                  └─ stored as module.weight_scale (float)

                      FAKE QUANTIZE (forward pass)
weight (bf16) ──→ x/scale ──→ round+clamp ──→ ×scale ──→ weight_qdq (bf16)
                  (float)      (fp8 range)     (float)    (with quant error)

                       COMPRESS (save)
weight (bf16) ──→ quantize(w/scale) ──→ weight (float8_e4m3fn)  → safetensors
scale (float) ──→ 127+log2(scale)   ──→ scale (uint8 E8M0)     → safetensors

                     DECOMPRESS (load)
weight (fp8)  ──→ weight.to(float)  ──→ ×scale ──→ weight (float)
scale (uint8) ──→ 2^(uint8-127)     ──→ scale (float)
```
