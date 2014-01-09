"""Utility I/O functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json

import numpy as np


# -----------------------------------------------------------------------------
# Data type conversions
# -----------------------------------------------------------------------------
_maxint16 = 2.**15-1
_maxint16inv = 1./_maxint16
_dtype_factors = {
    (np.int16, np.float32): _maxint16inv,
    (np.int16, np.float64): _maxint16inv,
    (np.float32, np.int16): _maxint16,
    (np.float64, np.int16): _maxint16,
}

def convert_dtype(data, dtype=None):
    if not dtype:
        return data
    dtype_old = data.dtype
    if dtype_old == dtype:
        return data
    data_new = data.astype(dtype)
    factor = _dtype_factors.get((dtype_old, dtype), 1)
    # We avoid unnecessary array copy when factor == 1
    if factor != 1:
        return data_new * factor
    else:
        return data_new 


# -----------------------------------------------------------------------------
# JSON functions
# -----------------------------------------------------------------------------
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def save_json(path, d):
    with open(path, 'w') as f:
        json.dump(d, f, indent=4)

