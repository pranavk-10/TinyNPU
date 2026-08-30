import numpy as np


def float_to_int8(values, scale=127):
    values = np.clip(values, -1, 1)
    return np.round(values * scale).astype(np.int8)


def int8_to_hex(value):
    return f"{int(value) & 0xFF:02X}"


def export_mem(filename, array):

    flat_array = array.flatten()

    with open(filename, "w") as f:

        for value in flat_array:
            f.write(int8_to_hex(value) + "\n")


# Load trained parameters
weights1 = np.load("weights1.npy")
bias1 = np.load("bias1.npy")

weights2 = np.load("weights2.npy")
bias2 = np.load("bias2.npy")

weights3 = np.load("weights3.npy")
bias3 = np.load("bias3.npy")


# Convert to INT8
weights1_int8 = float_to_int8(weights1)
bias1_int8 = float_to_int8(bias1)

weights2_int8 = float_to_int8(weights2)
bias2_int8 = float_to_int8(bias2)

weights3_int8 = float_to_int8(weights3)
bias3_int8 = float_to_int8(bias3)


# Export Verilog memory files
export_mem("layer1_weights.mem", weights1_int8)
export_mem("layer1_bias.mem", bias1_int8)

export_mem("layer2_weights.mem", weights2_int8)
export_mem("layer2_bias.mem", bias2_int8)

export_mem("layer3_weights.mem", weights3_int8)
export_mem("layer3_bias.mem", bias3_int8)


print("Verilog memory files generated!")