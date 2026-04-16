# MXFP4 Decompression From Fresh `main`

## Summary

Implement MXFP4 decompression on top of a fresh branch created from `main`.
Keep the change set limited to standard MXFP4 support only. Do not add,
modify, or test any `RCEIL` behavior.

## Key Changes

- Start implementation from the current `main` branch in this repo and create a
  new feature branch before any edits.
- Add MXFP4 decompression in `MXFP4PackedCompressor`.
  - Reuse the existing packed-FP4 unpack path.
  - Decode MX `weight_scale` from stored E8M0 `uint8` values back to float with
    the existing MX scale helper.
  - Dequantize packed weights using the decoded scale and existing global scale
    handling.
  - Return dense `weight` and decoded `weight_scale` in the same shape/dtype
    contract used by the other decompressors.
- Leave format selection and preset definitions unchanged.
  - `mxfp4-pack-quantized` remains the format for FP4 with `group_size=32`.
  - No changes to config names, public API, or llm-compressor recipe shape.
- Limit tests to non-`RCEIL` MXFP4 behavior.
  - Remove the MXFP4 `xfail` in the module compress/decompress test and make
    the existing MXFP4 round-trip cases pass.
  - Add or update focused MXFP4 tests for normal scale decode and dense
    reconstruction only.
  - Do not add `RCEIL` test coverage or touch existing `RCEIL` utilities unless
    required by shared code safety.

## Test Plan

- Run the MXFP4 unit tests that cover compression/decompression round trips.
- Run nearby packed FP4 tests to confirm no regression in shared unpack/dequantize
  behavior.
- End-to-end verify with the sibling `llm-compressor` checkout:
  - Quantize and save a small model using
    `/home/yiliu7/workspace/llm-compressor/experimental/mxfp4/qwen3_mxfp4.py`.
  - Load the saved checkpoint with `transformers`.
  - Run short generation and confirm load succeeds and text is clearly
    non-corrupted.
- Exclude any `MXFP4_RCEIL` scenario from the acceptance checklist.

## Assumptions

- Existing `RCEIL` code stays untouched unless a minimal shared-code adjustment
  is unavoidable for the non-`RCEIL` fix.
- Success means MXFP4 models can be decompressed and used for inference through
  the normal load path without adding new format options.
