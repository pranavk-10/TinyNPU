import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "quantized"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

from load import load_dataset
from model import NeuralNetwork


INT8_MAX = 127
INT8_MIN = -128

INT32_MAX = 2147483647
INT32_MIN = -2147483648


# ============================================================
# Generic INT8 Quantization
# ============================================================

def quantize_int8(values):

    max_value = np.max(
        np.abs(values)
    )

    if max_value == 0:

        scale = 1.0

        quantized = np.zeros_like(
            values,
            dtype=np.int8
        )

        return quantized, scale


    scale = max_value / INT8_MAX


    quantized = np.round(
        values / scale
    )


    quantized = np.clip(
        quantized,
        INT8_MIN,
        INT8_MAX
    )


    quantized = quantized.astype(
        np.int8
    )


    return quantized, scale


# ============================================================
# INT32 Bias Quantization
# ============================================================
#
# If:
#
# real_input  = input_int8  * input_scale
# real_weight = weight_int8 * weight_scale
#
# then:
#
# real_accumulator_scale =
#
#       input_scale * weight_scale
#
#
# Therefore:
#
# bias_int32 =
#
#       bias_float /
#       (input_scale * weight_scale)
#
# ============================================================

def quantize_bias(
    bias,
    input_scale,
    weight_scale
):

    accumulator_scale = (
        input_scale *
        weight_scale
    )


    quantized = np.round(
        bias /
        accumulator_scale
    )


    quantized = np.clip(
        quantized,
        INT32_MIN,
        INT32_MAX
    )


    quantized = quantized.astype(
        np.int32
    )


    return (
        quantized,
        accumulator_scale
    )


# ============================================================
# Save NumPy Array
# ============================================================

def save_array(
    filename,
    array
):

    np.save(
        OUTPUT_DIR / filename,
        array
    )


# ============================================================
# Save Scale
# ============================================================

def save_scale(
    filename,
    scale
):

    with open(
        OUTPUT_DIR / filename,
        "w"
    ) as file:

        file.write(
            str(scale)
        )


# ============================================================
# Load Dataset
# ============================================================

dataset_file = INPUTS_DIR / "tinynpu_21_feature_dataset.csv"
if not dataset_file.exists():
    dataset_file = Path("tinynpu_21_feature_dataset.csv")

X, y = load_dataset(str(dataset_file))


# ============================================================
# Load Trained Model Parameters
# ============================================================

weights1 = np.load(
    OUTPUT_DIR / "weights1.npy" if (OUTPUT_DIR / "weights1.npy").exists() else "weights1.npy"
)

bias1 = np.load(
    OUTPUT_DIR / "bias1.npy" if (OUTPUT_DIR / "bias1.npy").exists() else "bias1.npy"
)

weights2 = np.load(
    OUTPUT_DIR / "weights2.npy" if (OUTPUT_DIR / "weights2.npy").exists() else "weights2.npy"
)

bias2 = np.load(
    OUTPUT_DIR / "bias2.npy" if (OUTPUT_DIR / "bias2.npy").exists() else "bias2.npy"
)

weights3 = np.load(
    OUTPUT_DIR / "weights3.npy" if (OUTPUT_DIR / "weights3.npy").exists() else "weights3.npy"
)

bias3 = np.load(
    OUTPUT_DIR / "bias3.npy" if (OUTPUT_DIR / "bias3.npy").exists() else "bias3.npy"
)


# ============================================================
# 1. INPUT QUANTIZATION
# ============================================================

X_int8, input_scale = quantize_int8(
    X
)


save_array(
    "inputs_int8.npy",
    X_int8
)


save_scale(
    "input_scale.txt",
    input_scale
)


# ============================================================
# 2. WEIGHT QUANTIZATION
# ============================================================

weights1_int8, weights1_scale = (
    quantize_int8(weights1)
)


weights2_int8, weights2_scale = (
    quantize_int8(weights2)
)


weights3_int8, weights3_scale = (
    quantize_int8(weights3)
)


save_array(
    "weights1_int8.npy",
    weights1_int8
)

save_array(
    "weights2_int8.npy",
    weights2_int8
)

save_array(
    "weights3_int8.npy",
    weights3_int8
)


save_scale(
    "weights1_scale.txt",
    weights1_scale
)

save_scale(
    "weights2_scale.txt",
    weights2_scale
)

save_scale(
    "weights3_scale.txt",
    weights3_scale
)


# ============================================================
# 3. CALIBRATE ACTIVATION RANGES
# ============================================================
#
# We use the trained FP32 model to observe the hidden
# activation ranges.
#
# Layer 1:
#
#     input → W1 → ReLU → h1
#
# Layer 2:
#
#     h1 → W2 → ReLU → h2
#
# ============================================================


model = NeuralNetwork()


model.weights1 = weights1.copy()
model.bias1 = bias1.copy()

model.weights2 = weights2.copy()
model.bias2 = bias2.copy()

model.weights3 = weights3.copy()
model.bias3 = bias3.copy()


all_h1 = []
all_h2 = []
all_outputs = []


for sample in X:

    output, h1, h2, z1, z2, z3 = (
        model.forward(sample)
    )

    all_h1.append(h1)
    all_h2.append(h2)
    all_outputs.append(output[0])


all_h1 = np.array(
    all_h1,
    dtype=np.float32
)

all_h2 = np.array(
    all_h2,
    dtype=np.float32
)

all_outputs = np.array(
    all_outputs,
    dtype=np.float32
)


# ============================================================
# 4. ACTIVATION QUANTIZATION
# ============================================================

h1_int8, h1_scale = quantize_int8(
    all_h1
)

h2_int8, h2_scale = quantize_int8(
    all_h2
)


save_array(
    "activation1_int8.npy",
    h1_int8
)

save_array(
    "activation2_int8.npy",
    h2_int8
)


save_scale(
    "activation1_scale.txt",
    h1_scale
)

save_scale(
    "activation2_scale.txt",
    h2_scale
)


# ============================================================
# 5. BIAS QUANTIZATION
# ============================================================
#
# IMPORTANT:
#
# Layer 1 receives the original input.
#
# Layer 2 receives activation1.
#
# Layer 3 receives activation2.
#
# Therefore each bias uses the scale of its actual
# input activation.
# ============================================================


bias1_int32, bias1_scale = quantize_bias(
    bias1,
    input_scale,
    weights1_scale
)


bias2_int32, bias2_scale = quantize_bias(
    bias2,
    h1_scale,
    weights2_scale
)


bias3_int32, bias3_scale = quantize_bias(
    bias3,
    h2_scale,
    weights3_scale
)


save_array(
    "bias1_int32.npy",
    bias1_int32
)

save_array(
    "bias2_int32.npy",
    bias2_int32
)

save_array(
    "bias3_int32.npy",
    bias3_int32
)


save_scale(
    "bias1_scale.txt",
    bias1_scale
)

save_scale(
    "bias2_scale.txt",
    bias2_scale
)

save_scale(
    "bias3_scale.txt",
    bias3_scale
)


# ============================================================
# 6. SAVE FP32 OUTPUT REFERENCE
# ============================================================
#
# This is our golden reference.
#
# Later:
#
# Python FP32
#       ↓
# Expected probability
#
# RTL
#       ↓
# Hardware result
#
# We compare them.
# ============================================================


save_array(
    "outputs_fp32.npy",
    all_outputs
)


# ============================================================
# 7. SAVE TARGETS
# ============================================================

targets_int8 = y.astype(
    np.int8
)


save_array(
    "targets_int8.npy",
    targets_int8
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("============================================")
print("TinyNPU Full Quantization")
print("============================================")


print()
print("INPUTS")

print(
    "Shape:",
    X.shape
)

print(
    "INT8 range:",
    X_int8.min(),
    "to",
    X_int8.max()
)

print(
    "Scale:",
    input_scale
)


print()
print("WEIGHTS")


print(
    "W1:",
    weights1.shape
)

print(
    "INT8 range:",
    weights1_int8.min(),
    "to",
    weights1_int8.max()
)

print(
    "Scale:",
    weights1_scale
)


print(
    "W2:",
    weights2.shape
)

print(
    "INT8 range:",
    weights2_int8.min(),
    "to",
    weights2_int8.max()
)

print(
    "Scale:",
    weights2_scale
)


print(
    "W3:",
    weights3.shape
)

print(
    "INT8 range:",
    weights3_int8.min(),
    "to",
    weights3_int8.max()
)

print(
    "Scale:",
    weights3_scale
)


print()
print("ACTIVATIONS")


print(
    "Activation 1:",
    "INT8 range:",
    h1_int8.min(),
    "to",
    h1_int8.max()
)

print(
    "Activation 1 scale:",
    h1_scale
)


print(
    "Activation 2:",
    "INT8 range:",
    h2_int8.min(),
    "to",
    h2_int8.max()
)

print(
    "Activation 2 scale:",
    h2_scale
)


print()
print("BIASES")

print(
    "Bias 1 INT32:",
    bias1_int32
)

print(
    "Bias 2 INT32:",
    bias2_int32
)

print(
    "Bias 3 INT32:",
    bias3_int32
)


print()
print("OUTPUTS")

print(
    "FP32 probabilities:",
    all_outputs
)


print()
print("TARGETS")

print(
    targets_int8
)


print()
print("============================================")
print("FULL QUANTIZATION COMPLETE!")
print("============================================")

print()
print("Generated files:")
print()

print("INPUTS")
print("  inputs_int8.npy")
print("  input_scale.txt")

print()

print("WEIGHTS")
print("  weights1_int8.npy")
print("  weights1_scale.txt")
print("  weights2_int8.npy")
print("  weights2_scale.txt")
print("  weights3_int8.npy")
print("  weights3_scale.txt")

print()

print("ACTIVATIONS")
print("  activation1_int8.npy")
print("  activation1_scale.txt")
print("  activation2_int8.npy")
print("  activation2_scale.txt")

print()

print("BIASES")
print("  bias1_int32.npy")
print("  bias1_scale.txt")
print("  bias2_int32.npy")
print("  bias2_scale.txt")
print("  bias3_int32.npy")
print("  bias3_scale.txt")

print()

print("REFERENCE")
print("  outputs_fp32.npy")
print("  targets_int8.npy")