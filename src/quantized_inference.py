import numpy as np
from pathlib import Path


# ============================================================
# TinyNPU - Quantized INT8 Inference Reference
# ============================================================
#
# Network:
#
#                 21 → 14 → 7 → 1
#
# Arithmetic:
#
#     INT8 input
#       ×
#     INT8 weights
#       ↓
#     INT32 accumulator
#       +
#     INT32 bias
#       ↓
#     Requantize to INT8
#       ↓
#     ReLU
#       ↓
#     Next layer
#
# Final layer:
#
#     INT32 accumulator
#          ↓
#     Convert to FP32
#          ↓
#       Sigmoid
#          ↓
#     Probability
#
# This is the Python reference model that
# our future RTL implementation will match.
# ============================================================


QUANTIZED_DIR = Path("quantized")

INT8_MAX = 127
INT8_MIN = -128


# ============================================================
# Load Scale
# ============================================================

def load_scale(filename):

    with open(
        QUANTIZED_DIR / filename,
        "r"
    ) as file:

        return float(file.read())


# ============================================================
# Load Quantized Inputs
# ============================================================

inputs_int8 = np.load(
    QUANTIZED_DIR / "inputs_int8.npy"
)


# ============================================================
# Load Quantized Weights
# ============================================================

weights1_int8 = np.load(
    QUANTIZED_DIR / "weights1_int8.npy"
)

weights2_int8 = np.load(
    QUANTIZED_DIR / "weights2_int8.npy"
)

weights3_int8 = np.load(
    QUANTIZED_DIR / "weights3_int8.npy"
)


# ============================================================
# Load INT32 Biases
# ============================================================

bias1_int32 = np.load(
    QUANTIZED_DIR / "bias1_int32.npy"
)

bias2_int32 = np.load(
    QUANTIZED_DIR / "bias2_int32.npy"
)

bias3_int32 = np.load(
    QUANTIZED_DIR / "bias3_int32.npy"
)


# ============================================================
# Load Targets
# ============================================================

targets = np.load(
    QUANTIZED_DIR / "targets_int8.npy"
)


# ============================================================
# Load FP32 Reference Outputs
# ============================================================

fp32_outputs = np.load(
    QUANTIZED_DIR / "outputs_fp32.npy"
)


# ============================================================
# Load Scales
# ============================================================

input_scale = load_scale(
    "input_scale.txt"
)

weights1_scale = load_scale(
    "weights1_scale.txt"
)

weights2_scale = load_scale(
    "weights2_scale.txt"
)

weights3_scale = load_scale(
    "weights3_scale.txt"
)

activation1_scale = load_scale(
    "activation1_scale.txt"
)

activation2_scale = load_scale(
    "activation2_scale.txt"
)


# ============================================================
# Requantize INT32 → INT8
# ============================================================

def requantize_int32(
    accumulator,
    accumulator_scale,
    output_scale
):

    # Convert INT32 accumulator into real value

    real_value = (
        accumulator *
        accumulator_scale
    )


    # Convert real value to INT8

    quantized = np.round(
        real_value /
        output_scale
    )


    # Keep inside INT8 range

    quantized = np.clip(
        quantized,
        INT8_MIN,
        INT8_MAX
    )


    return quantized.astype(
        np.int8
    )


# ============================================================
# INT8 ReLU
# ============================================================

def relu_int8(values):

    return np.maximum(
        values,
        0
    ).astype(
        np.int8
    )


# ============================================================
# Sigmoid
# ============================================================

def sigmoid(value):

    value = np.clip(
        value,
        -50,
        50
    )

    return 1.0 / (
        1.0 +
        np.exp(-value)
    )


# ============================================================
# QUANTIZED FORWARD PASS
# ============================================================

def quantized_forward(input_vector):


    # ========================================================
    # LAYER 1
    # ========================================================
    #
    # 21 inputs
    #     ↓
    # INT8 × INT8
    #     ↓
    # INT32 accumulation
    #     ↓
    # INT32 bias
    #     ↓
    # INT8
    #     ↓
    # ReLU
    # ========================================================

    accumulator1 = (
        np.dot(
            weights1_int8.astype(
                np.int32
            ),
            input_vector.astype(
                np.int32
            )
        )
        +
        bias1_int32
    )


    accumulator1_scale = (
        input_scale *
        weights1_scale
    )


    h1_int8 = requantize_int32(
        accumulator1,
        accumulator1_scale,
        activation1_scale
    )


    h1_int8 = relu_int8(
        h1_int8
    )


    # ========================================================
    # LAYER 2
    # ========================================================
    #
    # 14 inputs
    #     ↓
    # INT8 × INT8
    #     ↓
    # INT32 accumulation
    #     ↓
    # INT32 bias
    #     ↓
    # INT8
    #     ↓
    # ReLU
    # ========================================================

    accumulator2 = (
        np.dot(
            weights2_int8.astype(
                np.int32
            ),
            h1_int8.astype(
                np.int32
            )
        )
        +
        bias2_int32
    )


    accumulator2_scale = (
        activation1_scale *
        weights2_scale
    )


    h2_int8 = requantize_int32(
        accumulator2,
        accumulator2_scale,
        activation2_scale
    )


    h2_int8 = relu_int8(
        h2_int8
    )


    # ========================================================
    # LAYER 3
    # ========================================================
    #
    # 7 inputs
    #     ↓
    # INT8 × INT8
    #     ↓
    # INT32 accumulation
    #     ↓
    # INT32 bias
    #     ↓
    # FP32
    #     ↓
    # Sigmoid
    # ========================================================

    accumulator3 = (
        np.dot(
            weights3_int8.astype(
                np.int32
            ),
            h2_int8.astype(
                np.int32
            )
        )
        +
        bias3_int32
    )


    accumulator3_scale = (
        activation2_scale *
        weights3_scale
    )


    # Convert final accumulator to FP32

    z3 = (
        accumulator3 *
        accumulator3_scale
    )


    # Sigmoid

    output = sigmoid(
        z3[0]
    )


    return (
        output,
        h1_int8,
        h2_int8,
        accumulator1,
        accumulator2,
        accumulator3
    )


# ============================================================
# RUN QUANTIZED INFERENCE
# ============================================================

print()
print("============================================")
print("TinyNPU Quantized INT8 Inference")
print("============================================")


correct = 0


for i in range(
    len(inputs_int8)
):


    (
        quantized_output,
        h1,
        h2,
        acc1,
        acc2,
        acc3
    ) = quantized_forward(
        inputs_int8[i]
    )


    predicted_class = (
        1
        if quantized_output >= 0.5
        else 0
    )


    target = int(
        targets[i]
    )


    fp32_output = float(
        fp32_outputs[i]
    )


    if predicted_class == target:

        correct += 1


    print()
    print(
        f"Sample {i + 1}"
    )

    print(
        f"Target             : "
        f"{target}"
    )

    print(
        f"FP32 probability   : "
        f"{fp32_output:.6f}"
    )

    print(
        f"INT8 probability   : "
        f"{quantized_output:.6f}"
    )

    print(
        f"Prediction         : "
        f"{predicted_class}"
    )

    print(
        f"Hidden 1 INT8      : "
        f"{h1}"
    )

    print(
        f"Hidden 2 INT8      : "
        f"{h2}"
    )


# ============================================================
# ACCURACY
# ============================================================

accuracy = (
    correct /
    len(inputs_int8)
) * 100


print()
print("============================================")
print("Quantized Model Accuracy")
print("============================================")

print(
    f"Correct: "
    f"{correct}/{len(inputs_int8)}"
)

print(
    f"Accuracy: "
    f"{accuracy:.2f}%"
)


# ============================================================
# SCALE INFORMATION
# ============================================================

print()
print("============================================")
print("Scale Information")
print("============================================")

print(
    "Input scale          :",
    input_scale
)

print(
    "W1 scale             :",
    weights1_scale
)

print(
    "Activation 1 scale   :",
    activation1_scale
)

print(
    "W2 scale             :",
    weights2_scale
)

print(
    "Activation 2 scale   :",
    activation2_scale
)

print(
    "W3 scale             :",
    weights3_scale
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================")
print("INT8 inference complete!")
print("============================================")