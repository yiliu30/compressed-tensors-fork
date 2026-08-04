# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.quantization.utils.ue5m3_utils import (
    cast_to_ue5m3,
    float_to_ue5m3_bits,
    ue5m3_bits_to_float,
)


def test_pack_unpack():
    x = torch.Tensor(
        [
            [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000, -0.0000],
            [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000, 0.0000],
            [-3.0000, -6.0000, -0.5000, -2.0000, -0.5000, -1.5000, -0.0000, -0.0000],
            [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000, 0.0000],
        ]
    )

    dense_dtype = torch.bfloat16
    x = x.to(dense_dtype)
    m, n = x.shape
    packed = pack_fp4_to_uint8(x)
    assert packed.dtype == torch.uint8
    unpacked = unpack_fp4_from_uint8(packed, m, n, dtype=dense_dtype)
    assert unpacked.dtype == dense_dtype

    assert torch.equal(unpacked, x)
    sign_bitx = torch.signbit(x)
    sign_bitout = torch.signbit(unpacked)
    assert torch.equal(sign_bitout, sign_bitx)


def test_pack_unpack_odd_dims():
    x = torch.Tensor(
        [
            [-0.5000, -6.0000, -0.5000, -1.5000, -1.0000, 6.0000, 0.0000],
            [-1.0000, -6.0000, -0.5000, -0.0000, 0.5000, 0.5000, -0.0000],
            [1.5000, 6.0000, -0.0000, -0.5000, 1.0000, 1.0000, -0.0000],
        ]
    )

    with pytest.raises((ValueError, torch._dynamo.exc.Unsupported)):
        _ = pack_fp4_to_uint8(x)


def test_compress_scale_without_scale_dtype():
    scale = torch.randn(10, dtype=torch.bfloat16)
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        group_size=16,
    )

    compressed_scale = NVFP4PackedCompressor._compress_scale(scale, quant_args)
    assert compressed_scale.dtype == torch.float8_e4m3fn


def test_ue5m3_round_trip_reference_values():
    values = torch.tensor(
        [0.0, 2.0**-17, 2.0**-14, 0.1, 1.0, 3.25, 1000.0, 114688.0, -1.0, float("inf")],
        dtype=torch.float32,
    )

    bits = float_to_ue5m3_bits(values)
    decoded = ue5m3_bits_to_float(bits)

    assert bits.dtype == torch.uint8
    assert decoded[0].item() == 0.0
    assert math.isclose(decoded[1].item(), 2.0**-17, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(decoded[2].item(), 2.0**-14, rel_tol=0.0, abs_tol=0.0)
    assert decoded[8].item() == 0.0
    assert decoded[9].item() == 114688.0


def test_cast_to_ue5m3_matches_encode_decode():
    values = torch.rand(64, dtype=torch.float32) * 1024
    assert torch.equal(
        cast_to_ue5m3(values),
        ue5m3_bits_to_float(float_to_ue5m3_bits(values)),
    )


def test_compress_scale_with_ue5m3_scale_format():
    scale = torch.tensor([2.0**-17, 0.1, 1.0, 32.0, 114688.0], dtype=torch.float32)
    quant_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        symmetric=True,
        strategy="tensor_group",
        group_size=16,
        scale_format="ue5m3",
    )

    compressed_scale = NVFP4PackedCompressor._compress_scale(scale, quant_args)
    restored_scale = NVFP4PackedCompressor._decompress_scale(
        compressed_scale, torch.float32, quant_args
    )

    assert compressed_scale.dtype == torch.uint8
    assert torch.equal(restored_scale, cast_to_ue5m3(scale))


def test_nvfp4_ue5m3_weight_round_trip():
    weight = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16)
    scale = torch.tensor([[0.5]], dtype=torch.float32)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="float",
            strategy="tensor_group",
            group_size=16,
            scale_format="ue5m3",
        ),
    )

    compressed = NVFP4PackedCompressor.compress(
        {"weight": weight, "weight_scale": scale}, scheme
    )
    decompressed = NVFP4PackedCompressor.decompress(compressed, scheme)

    assert compressed["weight_scale"].dtype == torch.uint8
    assert decompressed["weight_scale"].dtype == torch.float32
    assert decompressed["weight"].shape == weight.shape
