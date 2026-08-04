# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from compressed_tensors.quantization.quant_args import UE5M3_DATA


__all__ = [
    "cast_to_ue5m3",
    "float_to_ue5m3_bits",
    "ue5m3_bits_to_float",
]


def float_to_ue5m3_bits(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.to(torch.float32), nan=UE5M3_DATA.max)
    x = torch.clamp(x, min=0.0, max=UE5M3_DATA.max)

    bits = torch.zeros_like(x, dtype=torch.uint8)
    nonzero_mask = x > 0
    if not torch.any(nonzero_mask):
        return bits

    x_nonzero = x[nonzero_mask]
    normal_mask = x_nonzero >= UE5M3_DATA.min_normal
    out_vals = torch.zeros_like(x_nonzero, dtype=torch.uint8)

    if torch.any(normal_mask):
        x_normal = x_nonzero[normal_mask]
        mantissa, exponent = torch.frexp(x_normal)
        m3 = torch.clamp(torch.round((mantissa - 0.5) * 16), 0, 7).to(torch.uint8)
        e5 = torch.clamp(exponent + 14, 0, 31).to(torch.uint8)
        out_vals[normal_mask] = (e5 << 3) | m3

    if torch.any(~normal_mask):
        x_subnormal = x_nonzero[~normal_mask]
        m_sub = torch.clamp(
            torch.round(x_subnormal / UE5M3_DATA.min_normal * 8), 1, 7
        ).to(torch.uint8)
        out_vals[~normal_mask] = m_sub

    bits[nonzero_mask] = out_vals
    return bits


def ue5m3_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    if bits.dtype != torch.uint8:
        raise TypeError(f"UE5M3 payload must be uint8, got {bits.dtype}")

    values = torch.zeros_like(bits, dtype=torch.float32)
    nonzero_mask = bits != 0
    if not torch.any(nonzero_mask):
        return values

    encoded = bits[nonzero_mask]
    exponent = ((encoded >> 3) & 0x1F).to(torch.int32)
    mantissa = (encoded & 0x07).to(torch.int32)

    is_nan = (exponent == 31) & (mantissa == 7)
    is_subnormal = exponent == 0
    is_normal = (exponent > 0) & (~is_nan)

    decoded = torch.zeros_like(exponent, dtype=torch.float32)
    decoded[is_subnormal] = (
        mantissa[is_subnormal].to(torch.float32) / 8.0
    ) * UE5M3_DATA.min_normal
    if torch.any(is_normal):
        mant = 1.0 + mantissa[is_normal].to(torch.float32) / 8.0
        exp = exponent[is_normal] - 15
        decoded[is_normal] = torch.ldexp(mant, exp)
    decoded[is_nan] = float("nan")

    values[nonzero_mask] = decoded
    return values


def cast_to_ue5m3(x: torch.Tensor) -> torch.Tensor:
    original_dtype = x.dtype
    return ue5m3_bits_to_float(float_to_ue5m3_bits(x)).to(original_dtype)
