import pandas as pd
import numpy as np


def load_dataset(file_path):

    # Load CSV
    data = pd.read_csv(file_path)

    # Separate input features and target
    X = data.drop("target", axis=1)
    y = data["target"]

    # Convert to NumPy arrays
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)

    return X, y