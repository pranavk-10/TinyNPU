import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self):

        # 21 → 14
        self.weights1 = np.random.randn(14, 21) * 0.1
        self.bias1 = np.zeros(14)

        # 14 → 7
        self.weights2 = np.random.randn(7, 14) * 0.1
        self.bias2 = np.zeros(7)

        # 7 → 1
        self.weights3 = np.random.randn(1, 7) * 0.1
        self.bias3 = np.zeros(1)


    def forward(self, inputs):

        # Layer 1
        z1 = np.dot(self.weights1, inputs) + self.bias1
        h1 = relu(z1)

        # Layer 2
        z2 = np.dot(self.weights2, h1) + self.bias2
        h2 = relu(z2)

        # Output layer
        z3 = np.dot(self.weights3, h2) + self.bias3
        output = sigmoid(z3)

        return output, h1, h2, z1, z2, z3