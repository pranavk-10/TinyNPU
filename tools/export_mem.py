import numpy as np
from pathlib import Path

# ============================================================
# TinyNPU - Export Quantized Data to .mem & .vh
# ============================================================
#
# Converts NumPy integer arrays into hexadecimal memory files
# that can later be loaded by Verilog/SystemVerilog using:
#
#     $readmemh()
#
# Also computes and exports fixed-point requantization
# parameters for SystemVerilog testbenches and top-level modules.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

QUANTIZED_DIR = DATA_DIR / "quantized" if (DATA_DIR / "quantized").exists() else Path("quantized")
MEM_DIR = DATA_DIR / "mem" if (DATA_DIR / "mem").exists() else Path("mem")

MEM_DIR.mkdir(
    parents=True,
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
# Compute & Export Fixed-Point Requantization Parameters
# ============================================================
#
# Layer 1: M1 = (input_scale * weights1_scale) / activation1_scale
# Layer 2: M2 = (activation1_scale * weights2_scale) / activation2_scale
# Layer 3: M3 = (activation2_scale * weights3_scale) / 1.0 (or output scale)
# ============================================================

def read_scale(filename):
    p = QUANTIZED_DIR / filename
    if p.exists():
        with open(p, "r") as f:
            return float(f.read().strip())
    return 1.0

input_scale = read_scale("input_scale.txt")
w1_scale = read_scale("weights1_scale.txt")
act1_scale = read_scale("activation1_scale.txt")

w2_scale = read_scale("weights2_scale.txt")
act2_scale = read_scale("activation2_scale.txt")

w3_scale = read_scale("weights3_scale.txt")

SHIFT_BITS = 20

m1_ratio = (input_scale * w1_scale) / act1_scale if act1_scale != 0 else 1.0
l1_multiplier = int(round(m1_ratio * (1 << SHIFT_BITS)))

m2_ratio = (act1_scale * w2_scale) / act2_scale if act2_scale != 0 else 1.0
l2_multiplier = int(round(m2_ratio * (1 << SHIFT_BITS)))

# For layer 3, binary decision is sign of accumulator (acc3 >= 0).
# A nominal scale allows logit/activation estimation.
m3_ratio = (act2_scale * w3_scale)
l3_multiplier = int(round(m3_ratio * (1 << SHIFT_BITS)))

inputs_data = np.load(QUANTIZED_DIR / "inputs_int8.npy")

vh_path = MEM_DIR / "tinynpu_params.vh"
with open(vh_path, "w") as vh_file:
    vh_file.write("// ============================================================\n")
    vh_file.write("// TinyNPU Auto-Generated Fixed-Point Requantization Parameters\n")
    vh_file.write("// ============================================================\n\n")
    vh_file.write(f"localparam integer TINYNPU_TOTAL_SAMPLES = {len(inputs_data)};\n\n")
    vh_file.write(f"localparam integer L1_MULTIPLIER = {l1_multiplier};\n")
    vh_file.write(f"localparam integer L1_SHIFT      = {SHIFT_BITS};\n\n")
    vh_file.write(f"localparam integer L2_MULTIPLIER = {l2_multiplier};\n")
    vh_file.write(f"localparam integer L2_SHIFT      = {SHIFT_BITS};\n\n")
    vh_file.write(f"localparam integer L3_MULTIPLIER = {l3_multiplier};\n")
    vh_file.write(f"localparam integer L3_SHIFT      = {SHIFT_BITS};\n")

print(f"{'tinynpu_params.vh':<25} Requantization parameters header generated")


# ============================================================
# Export Layer-by-Layer Golden Activations for RTL debugging
# ============================================================

w1_data = np.load(QUANTIZED_DIR / "weights1_int8.npy")
b1_data = np.load(QUANTIZED_DIR / "bias1_int32.npy")
w2_data = np.load(QUANTIZED_DIR / "weights2_int8.npy")
b2_data = np.load(QUANTIZED_DIR / "bias2_int32.npy")
w3_data = np.load(QUANTIZED_DIR / "weights3_int8.npy")
b3_data = np.load(QUANTIZED_DIR / "bias3_int32.npy")

l1_acts = []
l2_acts = []
l3_accs = []

for sample in inputs_data:
    acc1 = np.dot(w1_data.astype(np.int64), sample.astype(np.int64)) + b1_data
    h1 = np.clip((acc1 * l1_multiplier) >> SHIFT_BITS, -128, 127).astype(np.int8)
    h1 = np.maximum(0, h1)
    l1_acts.append(h1)

    acc2 = np.dot(w2_data.astype(np.int64), h1.astype(np.int64)) + b2_data
    h2 = np.clip((acc2 * l2_multiplier) >> SHIFT_BITS, -128, 127).astype(np.int8)
    h2 = np.maximum(0, h2)
    l2_acts.append(h2)

    acc3 = np.dot(w3_data.astype(np.int64), h2.astype(np.int64)) + b3_data
    l3_accs.append(acc3)

with open(MEM_DIR / "layer1_expected.mem", "w") as f:
    for act in np.array(l1_acts).flatten():
        f.write(int_to_hex(act, 8) + "\n")

with open(MEM_DIR / "layer2_expected.mem", "w") as f:
    for act in np.array(l2_acts).flatten():
        f.write(int_to_hex(act, 8) + "\n")

print(f"{'layer1_expected.mem':<25} {len(np.array(l1_acts).flatten())} values 8-bit")
print(f"{'layer2_expected.mem':<25} {len(np.array(l2_acts).flatten())} values 8-bit")


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================")
print("MEM export complete!")
print("============================================")

print()
print("Generated files inside data/mem/:")
print()
print("INPUT & PARAMS")
print("  inputs.mem")
print("  tinynpu_params.vh")
print()
print("LAYER 1")
print("  weights1.mem")
print("  bias1.mem")
print("  layer1_expected.mem")
print()
print("LAYER 2")
print("  weights2.mem")
print("  bias2.mem")
print("  layer2_expected.mem")
print()
print("LAYER 3")
print("  weights3.mem")
print("  bias3.mem")
print()
print("TEST & REFERENCE")
print("  targets.mem")
print("  expected_classes.mem")
print("  scales.txt")
print()
print("============================================")
print("Ready for Real-Model RTL Verification!")
print("============================================")