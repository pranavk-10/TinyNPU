import numpy as np


def train(
    model,
    X,
    y,
    learning_rate=0.01,
    epochs=1000
):

    for epoch in range(epochs):

        total_loss = 0


        for sample in range(len(X)):

            inputs = X[sample]
            target = y[sample]


            # ====================================================
            # FORWARD PASS
            # ====================================================

            output, h1, h2, z1, z2, z3 = model.forward(inputs)

            prediction = output[0]


            # ====================================================
            # BINARY CROSS-ENTROPY LOSS
            # ====================================================

            loss = -(
                target * np.log(prediction + 1e-8)
                +
                (1 - target) *
                np.log(1 - prediction + 1e-8)
            )

            total_loss += loss


            # ====================================================
            # BACKPROPAGATION
            # ====================================================

            # Output layer

            dz3 = prediction - target

            dW3 = dz3 * h2

            db3 = dz3


            # ====================================================
            # Layer 2
            # ====================================================

            dh2 = model.weights3[0] * dz3

            dz2 = dh2 * (z2 > 0)

            dW2 = np.outer(dz2, h1)

            db2 = dz2


            # ====================================================
            # Layer 1
            # ====================================================

            dh1 = np.dot(
                model.weights2.T,
                dz2
            )

            dz1 = dh1 * (z1 > 0)

            dW1 = np.outer(
                dz1,
                inputs
            )

            db1 = dz1


            # ====================================================
            # UPDATE PARAMETERS
            # ====================================================

            model.weights3 -= (
                learning_rate *
                dW3.reshape(1, 7)
            )

            model.bias3 -= (
                learning_rate *
                db3
            )


            model.weights2 -= (
                learning_rate *
                dW2
            )

            model.bias2 -= (
                learning_rate *
                db2
            )


            model.weights1 -= (
                learning_rate *
                dW1
            )

            model.bias1 -= (
                learning_rate *
                db1
            )


        # ====================================================
        # PRINT LOSS
        # ====================================================

        if epoch % 100 == 0:

            print(
                f"Epoch {epoch}, "
                f"Loss: "
                f"{total_loss / len(X):.6f}"
            )