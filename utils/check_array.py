import numpy as np


def check_array(X, dtype=None, ensure_2d=True):
    """检查输入数组的类型和形状"""
    X = np.asarray(X)

    if dtype:
        if not np.issubdtype(X.dtype, np.dtype(dtype)):
            raise ValueError(f"Expected array with dtype {dtype}, got {X.dtype}")

    if ensure_2d and X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")

    return X
