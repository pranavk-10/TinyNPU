import numpy as np

from load import load_dataset
from model import NeuralNetwork
from backprop import train


# ============================================================
# TinyNPU
# Neural Network:
#
#              21
#               ↓
#              14
#               ↓
#               7
#               ↓
#               1
#
# 21 → 14 → 7 → 1
# ============================================================


# ============================================================
# 1. LOAD DATASET
# ============================================================

X, y = load_dataset(
    "tinynpu_21_feature_dataset.csv"
)

print("Dataset loaded")

print(
    "Input shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)

print("================================")


# ============================================================
# 2. CREATE MODEL
# ============================================================

model = NeuralNetwork()


print("Neural Network Architecture:")
print("Input Layer  : 21")
print("Hidden Layer : 14")
print("Hidden Layer : 7")
print("Output Layer : 1")

print("================================")


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("Starting Training")
print("================================")

train(
    model,
    X,
    y,
    learning_rate=0.01,
    epochs=1000
)


# ============================================================
# 4. SAVE TRAINED FP32 PARAMETERS
# ============================================================

np.save(
    "weights1.npy",
    model.weights1
)

np.save(
    "bias1.npy",
    model.bias1
)


np.save(
    "weights2.npy",
    model.weights2
)

np.save(
    "bias2.npy",
    model.bias2
)


np.save(
    "weights3.npy",
    model.weights3
)

np.save(
    "bias3.npy",
    model.bias3
)


print("================================")
print("FP32 weights and biases exported!")
print("================================")


# ============================================================
# 5. FP32 MODEL PREDICTIONS
# ============================================================

print("FP32 Model Predictions")
print("================================")


for i in range(
    len(X)
):

    output, h1, h2, z1, z2, z3 = (
        model.forward(
            X[i]
        )
    )


    probability = float(
        output[0]
    )


    prediction = (
        1
        if probability >= 0.5
        else 0
    )


    print(
        f"Sample {i + 1}: "
        f"Target = {int(y[i])}, "
        f"Probability = {probability:.4f}, "
        f"Prediction = {prediction}"
    )


# ============================================================
# 6. RUN FULL QUANTIZATION
# ============================================================

print()
print("================================")
print("Starting Full Quantization")
print("================================")


import quantize_model


# ============================================================
# 7. COMPLETE
# ============================================================

print()
print("================================")
print("TinyNPU Pipeline Complete!")
print("================================")

print()
print("Pipeline:")
print()
print("CSV Dataset")
print("    ↓")
print("FP32 Training")
print("    ↓")
print("Save Weights + Biases")
print("    ↓")
print("FP32 Inference")
print("    ↓")
print("INT8 / INT32 Quantization")
print("    ↓")
print("Quantized Parameters")
print("    ↓")
print("Ready for Quantized Inference")
print("    ↓")
print("Ready for .mem Export")
print("    ↓")
print("Ready for RTL")