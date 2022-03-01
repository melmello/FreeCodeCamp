import numpy as np


def calculate(array):
    if len(array) != 9:
        raise ValueError("List must contain nine numbers.")
    np_array = np.array(array).reshape(3, 3)
    return {
        "mean": [np_array.mean(axis=0).tolist(), np_array.mean(axis=1).tolist(), np_array.mean()],
        "variance": [np_array.var(axis=0).tolist(), np_array.var(axis=1).tolist(), np_array.var()],
        "standard deviation": [np_array.std(axis=0).tolist(), np_array.std(axis=1).tolist(), np_array.std()],
        "max": [np_array.max(axis=0).tolist(), np_array.max(axis=1).tolist(), np_array.max()],
        "min": [np_array.min(axis=0).tolist(), np_array.min(axis=1).tolist(), np_array.min()],
        "sum": [np_array.sum(axis=0).tolist(), np_array.sum(axis=1).tolist(), np_array.sum()]
    }
