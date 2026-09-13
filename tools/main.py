import sys
from pathlib import Path

# Add project root and src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
DATA_QUANTIZED_DIR = PROJECT_ROOT / "data" / "quantized"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import numpy as np
from load import load_dataset
from model import NeuralNetwork
from backprop import train


# ============================================================
# TinyNPU
# Neural Network:
#
#              21
#               v
#              14
#               v
#               7
#               v
#               1
#
# 21 -> 14 -> 7 -> 1
# ============================================================


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset_file = DATA_INPUTS_DIR / "tinynpu_21_feature_dataset.csv"
if not dataset_file.exists():
    dataset_file = Path("tinynpu_21_feature_dataset.csv")

X, y = load_dataset(str(dataset_file))

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
    DATA_QUANTIZED_DIR / "weights1.npy",
    model.weights1
)

np.save(
    DATA_QUANTIZED_DIR / "bias1.npy",
    model.bias1
)


np.save(
    DATA_QUANTIZED_DIR / "weights2.npy",
    model.weights2
)

np.save(
    DATA_QUANTIZED_DIR / "bias2.npy",
    model.bias2
)


np.save(
    DATA_QUANTIZED_DIR / "weights3.npy",
    model.weights3
)

np.save(
    DATA_QUANTIZED_DIR / "bias3.npy",
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
# 6. RUN FULL QUANTIZATION & MEM EXPORT
# ============================================================

print()
print("================================")
print("Starting Full Quantization")
print("================================")

import quantize_model
import export_mem


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
print("    |")
print("    v")
print("FP32 Training")
print("    |")
print("    v")
print("Save Weights + Biases")
print("    |")
print("    v")
print("FP32 Inference")
print("    |")
print("    v")
print("INT8 / INT32 Quantization")
print("    |")
print("    v")
print("Quantized Parameters")
print("    |")
print("    v")
print("Quantized Inference Reference")
print("    |")
print("    v")
print(".mem Export for RTL Simulation")
print("    |")
print("    v")
print("Ready for RTL Verification!")