import numpy as np


def calculate(array):
    if len(array) != 9:
        raise ValueError("List must contain nine numbers.")
    list = np.array(array).reshape(3, 3)
    return {
        'mean': [list.mean(axis=0).tolist(), list.mean(axis=1).tolist(), list.mean()],
        'variance': [list.var(axis=0).tolist(), list.var(axis=1).tolist(), list.var()],
        'standard deviation': [list.std(axis=0).tolist(), list.std(axis=1).tolist(), list.std()],
        'max': [list.max(axis=0).tolist(), list.max(axis=1).tolist(), list.max()],
        'min': [list.min(axis=0).tolist(), list.min(axis=1).tolist(), list.min()],
        'sum': [list.sum(axis=0).tolist(), list.sum(axis=1).tolist(), list.sum()]
    }
