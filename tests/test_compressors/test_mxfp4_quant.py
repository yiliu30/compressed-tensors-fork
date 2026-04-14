# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
    ScaleCalculationMode,
)


def test_compress_scale_without_scale_dtype():
    """
    Test that MXFP4 compressor handles missing scale_dtype.

    (backward compatibility)
    """
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16).abs() + 1e-6  # Ensure positive values

    # Create QuantizationArgs without scale_dtype (as in older models)
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        # scale_dtype is not set (defaults to None)
    )

    # This should not raise an error and should default to uint8
    compressed_scale = MXFP4PackedCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype is uint8
    assert compressed_scale.dtype == torch.uint8


def test_compress_scale_with_scale_dtype():
    """Test that MXFP4 compressor respects explicit scale_dtype"""
    # Create a scale tensor
    scale = torch.randn(10, dtype=torch.bfloat16).abs() + 1e-6  # Ensure positive values

    # Create QuantizationArgs with explicit scale_dtype
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
    )

    # Compress the scale
    compressed_scale = MXFP4PackedCompressor._compress_scale(scale, quant_args)

    # Verify the output dtype matches the specified scale_dtype
    assert compressed_scale.dtype == torch.uint8


def test_compress_scale_with_rceil_mode():
    scale = torch.tensor([0.1, 0.4, 0.9, 6.0, 7.0], dtype=torch.float32)
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=32,
        scale_dtype=torch.uint8,
        scale_calculation_mode=ScaleCalculationMode.RCEIL,
    )

    compressed_scale = MXFP4PackedCompressor._compress_scale(scale, quant_args)

    expected = torch.tensor([124, 126, 127, 130, 130], dtype=torch.uint8)
    assert torch.equal(compressed_scale, expected)
