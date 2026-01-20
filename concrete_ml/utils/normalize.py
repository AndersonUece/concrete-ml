# concrete_ml/utils/normalize.py
import numpy as np

def minmax_scale(array: np.ndarray) -> np.ndarray:
    """Scale a numpy array to the [0,1] range."""
    min_val = array.min()
    max_val = array.max()
    if max_val == min_val:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)
