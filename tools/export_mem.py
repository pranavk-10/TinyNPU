import numpy as np
from pathlib import Path


# ============================================================
# TinyNPU - Export Quantized Data to .mem
# ============================================================
#
# Converts NumPy integer arrays into hexadecimal memory files
# that can later be loaded by Verilog/SystemVerilog using:
#
#     $readmemh()
#
#
# INT8:
#
#     -128 → 80
#       -1 → FF
#        0 → 00
#        1 → 01
#      127 → 7F
#
#
# INT32:
#
#     values are written as 8-digit two's-complement HEX.
# ============================================================


QUANTIZED_DIR = Path(
    "quantized"
)

MEM_DIR = Path(
    "mem"
)


# ============================================================
# Create mem directory
# ============================================================

MEM_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# Convert Integer to HEX
# ============================================================

def int_to_hex(
    value,
    bits
):

    # Create bit mask

    mask = (
        1 << bits
    ) - 1


    # Convert signed value into
    # two's-complement representation

    value = int(value) & mask


    # Number of hexadecimal digits

    width = bits // 4


    return f"{value:0{width}X}"


# ============================================================
# Export NumPy Array to .mem
# ============================================================

def export_mem(
    input_filename,
    output_filename,
    bits
):

    # Load NumPy array

    array = np.load(
        QUANTIZED_DIR /
        input_filename
    )


    # Flatten array

    flat_array = array.flatten()


    # Create .mem file

    with open(
        MEM_DIR /
        output_filename,
        "w"
    ) as file:


        for value in flat_array:

            file.write(
                int_to_hex(
                    value,
                    bits
                )
                +
                "\n"
            )


    print(
        f"{output_filename:<25}"
        f"{len(flat_array):>5} values "
        f"{bits}-bit"
    )


# ============================================================
# INPUTS
# ============================================================

export_mem(
    "inputs_int8.npy",
    "inputs.mem",
    8
)


# ============================================================
# LAYER 1
# ============================================================

export_mem(
    "weights1_int8.npy",
    "weights1.mem",
    8
)

export_mem(
    "bias1_int32.npy",
    "bias1.mem",
    32
)


# ============================================================
# LAYER 2
# ============================================================

export_mem(
    "weights2_int8.npy",
    "weights2.mem",
    8
)

export_mem(
    "bias2_int32.npy",
    "bias2.mem",
    32
)


# ============================================================
# LAYER 3
# ============================================================

export_mem(
    "weights3_int8.npy",
    "weights3.mem",
    8
)

export_mem(
    "bias3_int32.npy",
    "bias3.mem",
    32
)


# ============================================================
# TARGETS
# ============================================================

export_mem(
    "targets_int8.npy",
    "targets.mem",
    8
)


# ============================================================
# Expected Classification
# ============================================================
#
# FP32 probability:
#
#     >= 0.5 → class 1
#     <  0.5 → class 0
#
# These become our expected RTL results.
# ============================================================

outputs_fp32 = np.load(
    QUANTIZED_DIR /
    "outputs_fp32.npy"
)


expected_classes = (
    outputs_fp32 >= 0.5
).astype(
    np.int8
)


with open(
    MEM_DIR /
    "expected_classes.mem",
    "w"
) as file:


    for value in expected_classes:

        file.write(
            int_to_hex(
                value,
                8
            )
            +
            "\n"
        )


print(
    f"{'expected_classes.mem':<25}"
    f"{len(expected_classes):>5} values "
    f"8-bit"
)


# ============================================================
# Export Scales for Reference
# ============================================================
#
# Scales remain human-readable floating-point values.
# They are not directly used as hardware memory yet.
# ============================================================

scale_files = [

    "input_scale.txt",

    "weights1_scale.txt",
    "activation1_scale.txt",

    "weights2_scale.txt",
    "activation2_scale.txt",

    "weights3_scale.txt",

    "bias1_scale.txt",
    "bias2_scale.txt",
    "bias3_scale.txt"
]


with open(
    MEM_DIR /
    "scales.txt",
    "w"
) as output_file:


    for filename in scale_files:

        scale_path = (
            QUANTIZED_DIR /
            filename
        )


        if scale_path.exists():

            with open(
                scale_path,
                "r"
            ) as scale_file:

                value = (
                    scale_file
                    .read()
                    .strip()
                )


            output_file.write(
                f"{filename} = {value}\n"
            )


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================")
print("MEM export complete!")
print("============================================")

print()
print("Generated files inside mem/:")

print()

print("INPUT")
print("  inputs.mem")

print()

print("LAYER 1")
print("  weights1.mem")
print("  bias1.mem")

print()

print("LAYER 2")
print("  weights2.mem")
print("  bias2.mem")

print()

print("LAYER 3")
print("  weights3.mem")
print("  bias3.mem")

print()

print("TEST")
print("  targets.mem")
print("  expected_classes.mem")

print()

print("REFERENCE")
print("  scales.txt")

print()
print("============================================")
print("Ready for RTL!")
print("============================================")