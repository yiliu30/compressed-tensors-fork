# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.mx_utils import (
    compress_mx_scale,
    decompress_mx_scale,
)
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize
from compressed_tensors.utils import TensorStateDict


__all__ = ["MXFP4PackedCompressor"]


@BaseCompressor.register(name=CompressionFormat.mxfp4_pack_quantized.value)
class MXFP4PackedCompressor(NVFP4PackedCompressor):
    """
    Compressor for MXFP4 quantized models.

    Overrides scale compression to use log2 encoding (bias-127 exponent).
    """

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        scale_dtype = weights.scale_dtype or torch.uint8
        return compress_mx_scale(scale, scale_dtype)

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        """
        Decompress a per-module state dict.

        :param state_dict: local-name state dict (weight_packed, weight_scale, …)
        :param scheme: quantization scheme for the weight
        :return: decompressed state dict with weight in float dtype
        """
        state_dict = state_dict.copy()
        packed = state_dict.pop("weight_packed")
        scale = state_dict.get("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)

        m, n = packed.shape
        unpacked = unpack_fp4_from_uint8(packed, m, n * 2)

        scale_float = decompress_mx_scale(scale)

        state_dict["weight"] = dequantize(
            x_q=unpacked,
            scale=scale_float,
            global_scale=global_scale,
            dtype=unpacked.dtype,
        )
        state_dict["weight_scale"] = torch.nn.Parameter(
            scale_float, requires_grad=False
        )

        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """MXFP4 matches FP4 with group_size=32."""
        return (
            module_type == torch.nn.Linear
            and scheme.weights is not None
            and scheme.weights.num_bits == 4
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.group_size == 32
        )
