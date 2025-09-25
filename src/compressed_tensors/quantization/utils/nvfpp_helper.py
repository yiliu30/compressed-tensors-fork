import torch


import torch

torch.set_printoptions(precision=12)
import numpy as np


def show_bit_repr_in_format(s, width=32):
    # s = '01000000010100000000000000000000'
    # print it in groups of 4 bits, with header
    # header = " ".join([f"{i:>2}" for i in range(0, width, 4)])
    # print(header)
    print(" ".join([s[i : i + 4] for i in range(0, width, 4)]))


# np.binary_repr
def show_tenosr_binary(tensor, name=""):
    print(f"{name}:  ================>>>>>>>>>>>>>")
    np_bi = tensor.cpu().numpy()
    # print([np.binary_repr(x, width=32) for x in np_bi.flatten()])
    for v in np_bi.flatten():
        s = np.binary_repr(v, width=32)
        show_bit_repr_in_format(s)
        print()
    print(f"{name}:  ================<<<<<<<<<<<<<")


DATYE_FP8E5M3 = "fp8e5m3"

MANTISSA_BITS = {
    torch.float8_e5m2: 2,
    torch.float8_e5m2fnuz: 2,
    torch.float8_e4m3fn: 3,
    torch.float8_e4m3fnuz: 3,
    torch.float8_e8m0fnu: 0,
    DATYE_FP8E5M3: 3,
}

# As in np.finfo(dtype).minexp
MINEXP = {
    torch.float8_e5m2: -14,
    torch.float8_e5m2fnuz: -15,
    torch.float8_e4m3fn: -6,
    torch.float8_e4m3fnuz: -7,
    torch.float8_e8m0fnu: -127,
    DATYE_FP8E5M3: -15,
}
FLOAT8_DTYPES_WITH_INF = [torch.float8_e5m2]
from dataclasses import dataclass


@dataclass
class FINFO:
    max: int
    min: int


FINFOS = {
    DATYE_FP8E5M3: FINFO(max=114688, min=0),
}


def finfo(dtype, attr):
    try:
        return getattr(torch.finfo(dtype), attr)
    except:
        return FINFOS[dtype].__getattribute__(attr)


def simulate_fp8_precision(input, variant):
    """Round input (as float32) to the given float8 datatype variant."""

    # Constants
    dtype = torch.float32
    int_type = torch.int32
    mbits = MANTISSA_BITS[variant]
    minexp = MINEXP[variant]  # ml_dtypes.finfo(variant).

    input = input.to(dtype)

    # Extract bitfield components
    signs = torch.sign(input)
    input_int = torch.abs(input).view(int_type)

    # 0x7F800000
    #   |        |
    #   v        v
    #  0111 1111 1000 0000 0000 0000 0000 0000
    #  0000 0000 0111 0000 0000 0000 0000 0000

    exponent_bits = (input_int & 0x7F800000) >> 23
    mantissa_bits = input_int & 0x007FFFFF
    # exponent_bits - (127)
    exponent_base = exponent_bits - 0x7F

    # Add implicit leading 1 to mantissas, i.e. create 1.mmmmmmmm
    f32_is_normal = exponent_bits != 0
    mantissa_val_base = f32_is_normal * 0x00800000 + mantissa_bits

    # Shift mantissa to match minimum exponent - denormals in the lower
    # precision dtype remain normal in the higher precision dtype
    denormal_bits = torch.maximum(minexp - exponent_base, torch.tensor(0, dtype=int_type))
    mantissa_val = mantissa_val_base >> denormal_bits
    exponent = exponent_base + denormal_bits

    # Round off mantissas
    last_unrounded_bit = 1 << (23 - mbits)
    rounding_mask = last_unrounded_bit - 1
    mantissa_val_rounded = (mantissa_val + (rounding_mask >> 1)) & ~rounding_mask

    # Round ties to nearest even
    ties = (mantissa_val & rounding_mask) == (last_unrounded_bit >> 1)
    is_odd = (mantissa_val_rounded & last_unrounded_bit) != 0
    mantissa_val_rounded += (ties & is_odd) * last_unrounded_bit

    # Re-compose mantissa and exponent
    vals = (mantissa_val_rounded * 2.0 ** (-23 + exponent)).to(dtype)

    # Replace overflows with inf/NaN as appropriate (no saturation)
    have_inf = variant in FLOAT8_DTYPES_WITH_INF
    vals[
        vals
        >
        #  torch.finfo(variant).max
        finfo(variant, "max")
    ] = torch.inf if have_inf else torch.nan

    return vals * signs


import torch


def float_to_nvfpp_v1(x):
    """
    Convert floating-point tensor to E5M3 format (5 exponent bits, 3 mantissa bits)

    """
    assert not (x < 0).any(), "Only support non-negative values"
    # Initialize result tensor
    result = torch.zeros_like(x, dtype=torch.uint8)

    # Handle NaN values
    fp8e5m3_max = (2**16) * (1 + 1 / 2 + 1 / 4)
    nan_mask = (x > fp8e5m3_max) | torch.isnan(x)
    result[nan_mask] = 0xFF
    # Get bit representation
    x_bits = torch.zeros_like(x, dtype=torch.int32)
    x_bits.copy_(x.view(torch.int32))

    # Extract sign bit for later use
    sign_bit = x_bits & 0x80000000

    # Extract exponent and mantissa
    exp_mask = 0x7F800000
    mantissa_mask = 0x7FFFFF
    exponent = (x_bits & exp_mask) >> 23
    mantissa = x_bits & mantissa_mask

    # The minimum normal number in E5M3 is 2**(-14)
    # 00001 mmm
    #     ^
    is_denormal = x < (2 ** (-14))
    if is_denormal.any():
        x_bits_denormal = x_bits[is_denormal]
        denormal_mask = torch.zeros_like(x_bits_denormal, dtype=torch.int32)
        DENORM_MASK_VALUE = ((127 - 15) + (23 - 3) + 1) << 23
        denormal_mask.fill_(DENORM_MASK_VALUE)
        f_bits_denornal = x_bits_denormal.view(torch.float32) + denormal_mask.view(torch.float32)
        f_bits_denornal = f_bits_denornal.view(torch.int32) - DENORM_MASK_VALUE
        result[is_denormal] = f_bits_denornal.to(torch.uint8)
    normal_mask = ~(nan_mask | is_denormal)
    if normal_mask.any():
        x_bits = x_bits[normal_mask]
        mant_odd = (x_bits >> 20) & 1
        # Bits  31   30-23     22-0
        #      +---+----------+----------------------------+
        #      | S |EEE EEEE E|MMM MMMM MMMM MMMM MMMM MMMM|
        #      +---+----------+----------------------------+
        #                      001 0000 0000 0000 0000 0000  <- (x_bits >> 20) & 1
        #                          0111 1111 1111 1111 1111  <- 0x7FFFF
        x_bits += ((15 - 127) << 23) + 0x7FFFF
        x_bits += mant_odd
        x_bits_remove_right_zeros = x_bits >> 20
        result[normal_mask] = (x_bits_remove_right_zeros & 0xFF).to(torch.uint8)

    return result


# @torch.compile(fullgraph=True)
def float_to_nvfpp_v2(x):
    """
    Convert floating-point tensor to E5M3 format (5 exponent bits, 3 mantissa bits)

    """
    # assert not (x < 0).any(), "Only support non-negative values"
    # Initialize result tensor
    result = torch.zeros_like(x, dtype=torch.uint8)

    # Handle NaN values
    fp8e5m3_max = (2**16) * (1 + 1 / 2 + 1 / 4)
    nan_mask = (x > fp8e5m3_max) | torch.isnan(x)
    result[nan_mask] = 0xFF
    # Get bit representation
    x_bits = torch.zeros_like(x, dtype=torch.int32)
    x_bits.copy_(x.view(torch.int32))

    # Extract sign bit for later use
    sign_bit = x_bits & 0x80000000

    # Extract exponent and mantissa
    exp_mask = 0x7F800000
    mantissa_mask = 0x7FFFFF
    exponent = (x_bits & exp_mask) >> 23
    mantissa = x_bits & mantissa_mask

    # The minimum normal number in E5M3 is 2**(-14)
    # 00001 mmm
    #     ^
    # FIXME: (Yi) there are some issue for denormal value, need to fix it
    is_denormal = x < (2 ** (-14))
    denormal_mask = x < (2 ** (-14))
    denormal_res = x * denormal_mask

    _x_bits_denormal = x_bits * denormal_mask
    _denormal_mask = torch.zeros_like(_x_bits_denormal, dtype=torch.int32)
    DENORM_MASK_VALUE = ((127 - 15) + (23 - 3) + 1) << 23
    _denormal_mask.fill_(DENORM_MASK_VALUE)
    f_bits_denornal = _x_bits_denormal.view(torch.float32) + _denormal_mask.view(torch.float32)
    f_bits_denornal = f_bits_denornal.view(torch.int32) - DENORM_MASK_VALUE
    denormal_res = f_bits_denornal.to(torch.uint8)
    normal_mask = ~(nan_mask | is_denormal)
    # x_bits = x_bits[normal_mask]
    x_bits_normal = x_bits * normal_mask
    mant_odd = (x_bits_normal >> 20) & 1
    # Bits  31   30-23     22-0
    #      +---+----------+----------------------------+
    #      | S |EEE EEEE E|MMM MMMM MMMM MMMM MMMM MMMM|
    #      +---+----------+----------------------------+
    #                      001 0000 0000 0000 0000 0000  <- (x_bits >> 20) & 1
    #                          0111 1111 1111 1111 1111  <- 0x7FFFF
    x_bits_normal += ((15 - 127) << 23) + 0x7FFFF
    x_bits_normal += mant_odd
    x_bits_remove_right_zeros = x_bits_normal >> 20
    normal_res = (x_bits_remove_right_zeros & 0xFF).to(torch.uint8)
    final_res = denormal_res * denormal_mask + normal_res * normal_mask
    return final_res


import torch


def parse_e5m3(e5m3_tensor):
    """
    Convert E5M3 8-bit values to FP32 based on formula.

    Args:
        e5m3_tensor: torch.uint8 tensor of E5M3 values

    Returns:
        torch.float32 tensor of FP32 values
    """
    assert e5m3_tensor.dtype == torch.uint8, "Input tensor must be of type torch.uint8"
    e5m3_tensor = e5m3_tensor.to(torch.int32)

    # Extract components
    sign = torch.zeros_like(e5m3_tensor, dtype=torch.int32)
    exponent = (e5m3_tensor >> 3) & 0x1F  # 5-bit exponent
    mantissa = e5m3_tensor & 0x7  # 3-bit mantissa

    # Initialize result
    result = torch.zeros_like(e5m3_tensor, dtype=torch.float32)

    # Compute normal numbers (exponent != 0)
    normal_mask = exponent != 0
    if normal_mask.any():
        E = exponent[normal_mask]
        M = mantissa[normal_mask]
        # Convert mantissa bits to fraction: 1 + b0*2^-1 + b1*2^-2 + b2*2^-3
        frac = 1.0 + ((M >> 2) & 1) * 0.5 + ((M >> 1) & 1) * 0.25 + (M & 1) * 0.125
        result[normal_mask] = ((-1) ** sign[normal_mask]) * (2 ** (E.float() - 15)) * frac

    # Compute subnormal numbers (exponent == 0)
    subnorm_mask = exponent == 0
    if subnorm_mask.any():
        M = mantissa[subnorm_mask]
        # Fraction = b0*2^-1 + b1*2^-2 + b2*2^-3
        frac = ((M >> 2) & 1) * 0.5 + ((M >> 1) & 1) * 0.25 + (M & 1) * 0.125
        result[subnorm_mask] = ((-1) ** sign[subnorm_mask]) * (2 ** (-14)) * frac
    # check nan: 11111 111
    nan_mask = (exponent == 31) & (mantissa == 7)
    result[nan_mask] = float("nan")
    return result


F32_EXP_BIAS = 127
NVFPP_EXP_BIAS = 15


def nvfpp_to_float_v3(e5m3_data):
    """
    Convert E5M3 8-bit format to 32-bit float.

    Args:
        e5m3_data: Tensor of uint8 values in E5M3 format

    Returns:
        Tensor of float32 values
    """
    assert e5m3_data.dtype == torch.uint8, f"Input tensor must be of type torch.uint8, got {e5m3_data.dtype}"
    device = e5m3_data.device

    # upper part of the 32-bit word:
    #      +----+----+-----------------------------+
    #      |EEEEE|MMM|0000 0000 0000 0000 0000 0000|
    #      +----+----+-----------------------------+
    # Bits  31  26-30    16-25            0-15
    #      +---+-----+------------+-------------------+
    #      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
    #      +---+-----+------------+-------------------+
    #                     ^^^ ^^^^                      <- << 7
    #
    # Add 7 zeros -> start handle FP16
    # TODO: can we use to.(torch.float32)?
    temp = e5m3_data.to(torch.int32) << 7


# @torch.compile(fullgraph=True)
def nvfpp_to_float_v2(e5m3_data):
    """
    Convert E5M3 8-bit format to 32-bit float.

    Args:
        e5m3_data: Tensor of uint8 values in E5M3 format

    Returns:
        Tensor of float32 values
    """
    assert e5m3_data.dtype == torch.uint8, f"Input tensor must be of type torch.uint8, got {e5m3_data.dtype}"

    # upper part of the 32-bit word:
    #      +----+----+-----------------------------+
    #      |EEEEE|MMM|0000 0000 0000 0000 0000 0000|
    #      +----+----+-----------------------------+
    # Bits  31  26-30    16-25            0-15
    #      +---+-----+------------+-------------------+
    #      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
    #      +---+-----+------------+-------------------+
    #                     ^^^ ^^^^                      <- << 7
    #
    # Add 7 zeros -> start handle FP16
    # TODO: can we use to.(torch.float32)?
    temp = e5m3_data.to(torch.int32) << 7

    # Extract components
    # sign should always be 0 for nvfpp
    # Bits  15  10-14     0-9
    #      +---+-----+------------+
    #      | S |EEEEE|MM MMMM MMMM|
    #      +---+-----+------------+
    #                     ^^^ ^^^^ <-- New added 7 zeros
    #      sign| exp |  mantissa  |
    sign = (temp >> 15) & 0x1
    exp = (temp >> 10) & 0x1F
    mant = temp & 0x3FF

    # Initialize result tensor
    result = torch.zeros_like(e5m3_data, dtype=torch.float32)

    # Handle each case according to C++ implementation

    # Case 1: Zero (exp=0, mant=0)
    zero_mask = (exp == 0) & (mant == 0)
    result[zero_mask & (sign == 1)] = -0.0  # Apply sign to zeros

    # Case 2: Infinity/NaN (exp=31)
    # No Inf,
    # Nan mask 11111 111
    nan_mask = (exp == 31) & (((mant >> 7) & 0b111) == 0b111)
    result[nan_mask] = float("nan")

    # Case 3: Normal numbers
    # normal_mask = (exp != 0) & (exp != 31)
    normal_mask = (exp != 0) & ~nan_mask
    # TODO: remove the sign check
    # filter the normal value
    normal_val_exp = exp * normal_mask
    normal_val_sign = sign * normal_mask
    normal_val_mant = mant * normal_mask
    # Adjust bias to match FP32
    normal_exp_adjust = normal_val_exp + (F32_EXP_BIAS - NVFPP_EXP_BIAS)
    # Bits  31   30-23     22-0
    #      +---+----------+----------------------------+
    #      | S |EEE EEEE E|MMM MMMM MMMM MMMM MMMM MMMM|
    #      +---+----------+----------------------------+
    normal_val_in_bits = (normal_val_sign << 31) | (normal_exp_adjust << 23) | (normal_val_mant << 13)
    result_normal = normal_val_in_bits.view(torch.float32) * normal_mask
    # update the result
    # result[normal_mask] = result_normal[normal_mask]

    # Case 4: Denormals (exp=0, mant≠0)
    denorm_mask = (exp == 0) & (mant != 0)

    # 2^(-14) * (0.b1b2b3)
    denormal_val_mant = mant * denorm_mask
    # Fetch the first 3 bits of mantissa
    mantissa_fraction = (denormal_val_mant >> 7).float() / (2**3)
    result_denormal = (2 ** (-14)) * mantissa_fraction
    result_denormal = result_denormal * denorm_mask
    # below method is wrong, as FP32 add implicit leading 1 for normal number mantissa
    # denormal_val_exp = exp * denorm_mask
    # denormal_val_exp_adjust = denormal_val_exp +  (F32_EXP_BIAS - (NVFPP_EXP_BIAS-1))
    # denormal_val_mant = mant * denorm_mask
    # denormal_val_sign = sign * denorm_mask
    # denormal_val_in_bits = (denormal_val_sign << 31) | (denormal_val_exp_adjust << 23) | (denormal_val_mant << 13)
    # result_denormal = denormal_val_in_bits.view(torch.float32) * denorm_mask
    # result[denorm_mask] = result_denormal[denorm_mask]
    result = result_normal + result_denormal + result * (nan_mask | zero_mask)
    print(f"All posible E5M3 values:", result)
    return result


def nvfpp_to_float_v1(e5m3_data):
    """
    Convert E5M3 8-bit format to 32-bit float.

    Args:
        e5m3_data: Tensor of uint8 values in E5M3 format

    Returns:
        Tensor of float32 values
    """
    assert e5m3_data.dtype == torch.uint8, f"Input tensor must be of type torch.uint8, got {e5m3_data.dtype}"
    device = e5m3_data.device

    # upper part of the 32-bit word:
    #      +----+----+-----------------------------+
    #      |EEEEE|MMM|0000 0000 0000 0000 0000 0000|
    #      +----+----+-----------------------------+
    # Bits  31  26-30    16-25            0-15
    #      +---+-----+------------+-------------------+
    #      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
    #      +---+-----+------------+-------------------+
    #                     ^^^ ^^^^                      <- << 7
    #
    # Only add 7 zeros
    temp = e5m3_data.to(torch.int32) << 7

    # Extract components
    sign = (temp >> 15) & 0x1
    exp = (temp >> 10) & 0x1F
    mant = temp & 0x3FF

    # Initialize result tensor
    result = torch.zeros_like(e5m3_data, dtype=torch.float32)

    # Handle each case according to C++ implementation

    # Case 1: Zero (exp=0, mant=0)
    zero_mask = (exp == 0) & (mant == 0)
    result[zero_mask & (sign == 1)] = -0.0  # Apply sign to zeros

    # Case 2: Infinity/NaN (exp=31)
    inf_mask = 0
    # inf_mask = (exp == 31)
    # result[inf_mask & (sign == 0)] = float('inf')
    # result[inf_mask & (sign == 1)] = float('-inf')
    # nan mask 11111 111

    nan_mask = (exp == 31) & (((mant >> 7) & 0b111) == 0b111)
    result[nan_mask] = float("nan")
    # breakpoint()
    # Case 3: Normal numbers
    # normal_mask = (exp != 0) & (exp != 31)
    normal_mask = (exp != 0) & ~nan_mask
    # TODO: remove the sign check
    if normal_mask.any():
        normal_exp = exp[normal_mask] + (127 - 15)  # Adjust bias
        bits = (sign[normal_mask] << 31) | (normal_exp << 23) | (mant[normal_mask] << 13)
        result[normal_mask] = bits.view(torch.float32)

    # Case 4: Denormals (exp=0, mant≠0)
    denorm_mask = (exp == 0) & (mant != 0)
    # TODO: refine that denorm check
    if denorm_mask.any():
        for idx in torch.nonzero(denorm_mask, as_tuple=True)[0]:
            m = mant[idx].item()

            # Count leading zeros in 10-bit mantissa
            leading_zeros = 0
            test_bit = 0x200  # Start with bit 9 (highest bit in 10-bit mantissa)
            while test_bit > 0 and (m & test_bit) == 0:
                leading_zeros += 1
                test_bit >>= 1

            # Calculate adjusted exponent and mantissa as in C++ code
            e = 127 - 15 - leading_zeros
            m = (m << (leading_zeros + 1)) & 0x3FF
            s = sign[idx].item()

            # Create IEEE 754 bit pattern
            bits = (s << 31) | (e << 23) | (m << 13)
            result[idx] = torch.tensor([bits], dtype=torch.int32, device=device).view(torch.float32).item()

    return result


nvfpp_to_float = nvfpp_to_float_v2
# nvfpp_to_float = nvfpp_to_float_v3
float_to_nvfpp = float_to_nvfpp_v2


def test_all_possible_vals():
    import torch

    # Create all possible E5M3 values (0..255)
    e5m3_values = torch.arange(0, 256, dtype=torch.uint8)

    # Convert to FP32 using your function
    fp32_values = nvfpp_to_float(e5m3_values)

    # Print all values
    for e5m3, f32 in zip(e5m3_values.tolist(), fp32_values.tolist()):
        print(
            f"E5M3: {e5m3:#04x}  {e5m3:08b} -> FP32: {f32} | log2 {torch.log2(torch.tensor(f32)).item() if f32 > 0 else 'N/A'}"
        )

    diff_cnt = 0
    diff_index_lst = []
    fp32_hand = parse_e5m3(e5m3_values)
    fp32_sim = simulate_fp8_precision(fp32_hand, DATYE_FP8E5M3)

    for e5m3, f32, f32h in zip(e5m3_values.tolist(), fp32_values.tolist(), fp32_hand.tolist()):
        # if f32 != f32h:
        print(f" E5M3: {e5m3:#04x}  {e5m3:08b} -> FP32: {f32} vs {f32h}")
        # check nan
        if torch.isnan(torch.tensor(f32)) and torch.isnan(torch.tensor(f32h)):
            continue
        else:
            if f32 != f32h:
                diff_cnt += 1
                diff_index_lst.append(e5m3)
    print(f"Total mismatches: {diff_cnt} out of 256, {diff_index_lst}")


# test_all_possible_vals()
def test_convert_float_to_e5m3():
    import torch

    # Create all possible E5M3 values (0..255)
    e5m3_values = torch.arange(0, 256, dtype=torch.uint8)

    e5m2 = e5m3_values.view(torch.float8_e5m2)
    print(f"e5m2: {e5m2}")

    # Convert to FP32 using your function
    fp32_values = nvfpp_to_float(e5m3_values)

    fp32_values_excl_nan = fp32_values[~torch.isnan(fp32_values)]
    fp32_values_excl_nan = fp32_values_excl_nan[:]
    print(f"fp32_values_excl_nan: {fp32_values_excl_nan}")
    mid_val = (fp32_values_excl_nan[:-1] + fp32_values_excl_nan[1:]) / 2
    # mid_val = fp32_values_excl_nan
    print(f"mid_val: {mid_val}")
    mid_val_sim = simulate_fp8_precision(mid_val, DATYE_FP8E5M3)
    round_min_val_uint8 = float_to_nvfpp_v2(mid_val)
    print(f"round_min_val_uint8: {round_min_val_uint8}")
    round_min_val_uint8_fp32 = nvfpp_to_float(round_min_val_uint8)
    print(f"round_min_val_uint8_fp32: {round_min_val_uint8_fp32}")
    print(f"mid_val_sim: {mid_val_sim}")
    print(mid_val_sim == round_min_val_uint8_fp32)

    diff_cnt = 0
    for i in range(len(mid_val)):
        if mid_val_sim[i] != round_min_val_uint8_fp32[i]:
            diff = mid_val_sim[i] - round_min_val_uint8_fp32[i]
            diff_cnt += 1
            print(
                f"mid_val: {mid_val[i]} -> sim: {mid_val_sim[i]} vs round: {round_min_val_uint8_fp32[i]}, diff: {diff}"
            )
    # Print all va
    print(f"Total mismatches: {diff_cnt} out of {len(mid_val)}")


def bench_to_nvfpp():
    # Define shapes to benchmark
    shapes = [
        (1024,),  # 1K elements
        (32, 32),  # 1K elements, 2D
        (1024, 1024),  # 1M elements
        (4096, 4096),  # 1M elements
        (1024 * 128, 4096),  # 1M elements
        (8, 1024, 1024),  # 8M elements
        (16, 2048, 1024),  # 32M elements
    ]
    import torch
    import numpy as np
    from triton.testing import do_bench

    for shape in shapes:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)

        # Warm-up
        for _ in range(10):
            _ = float_to_nvfpp_v2(x)
            torch.cuda.synchronize()

        # Benchmark the optimized function
        optimized_time = do_bench(lambda: float_to_nvfpp_v2(x), rep=100)
        torch.cuda.synchronize()

        # Benchmark the original function
        original_time = do_bench(lambda: x.to(torch.float8_e4m3fn), rep=100)
        torch.cuda.synchronize()

        print(
            f"Shape: {shape}, Optimized Time: {optimized_time:.6f}s, Original Time: {original_time:.6f}s speedup: {original_time/optimized_time:.2f}x"
        )


# bench_to_nvfpp()

# test_convert_float_to_e5m3()
