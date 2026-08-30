from load import load_dataset
from model import NeuralNetwork
from backprop import train
import numpy as np


# Load data
X, y = load_dataset("tinynpu_21_feature_dataset.csv")


# Create neural network
model = NeuralNetwork()


# Train
train(
    model,
    X,
    y,
    learning_rate=0.01,
    epochs=1000
)

# Save trained weights and biases
np.save("weights1.npy", model.weights1)
np.save("bias1.npy", model.bias1)

np.save("weights2.npy", model.weights2)
np.save("bias2.npy", model.bias2)

np.save("weights3.npy", model.weights3)
np.save("bias3.npy", model.bias3)

print("Weights and biases exported!")

# Test
for i in range(len(X)):

    output, _, _, _, _, _ = model.forward(X[i])

    prediction = output[0]

    print(
        f"Target: {y[i]:.0f} "
        f"Prediction: {prediction:.4f}"
    )