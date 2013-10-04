"""Wrap dictionaries in Python objects for easy access of hierarchical structures."""
# -----------------------------------------------------------------------------
# Wrap functions
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# Wrap functions
# -----------------------------------------------------------------------------
class Wrapped(object):
    def __init__(self, d, index=None):
        self._d = d
        self._index = index
        
    def __getattr__(self, key):
        return wrap(self._d[key], self._index)
            
    def __getitem__(self, index):
        return wrap(self._d, index)
        
    def __dir__(self):
        return self._d.__dir__()

    def __repr__(self):
        return self._d.__repr__()
        
def wrap(d, index=None):
    if isinstance(d, dict):
        return Wrapped(d, index)
    elif isinstance(d, list):
        # Make sure the list is not empty.
        if not d:
            return d
        # Indexed.
        if index is not None:
            return d[index]
        # If it's a list of dict, wrap all the elements.
        if isinstance(d[0], dict):
            return map(wrap, d)
        # Otherwise, just return the list.
        else:
            return d
    elif isinstance(d, np.ndarray):
        if index is None:
            return d
        else:
            return d[index]
        
    if hasattr(d, '__iter__'):
        if index is None:
            return map(wrap, d)
        else:
            return d[index]
    else:
        if index is None:
            return d
        else:
            return d[index]
    