"""Wrap dictionaries in Python objects for easy access of hierarchical structures."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from six import iteritems


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def append(d, item):
    if isinstance(d, list):
        d.append(item)
        return d
    elif isinstance(d, np.ndarray):
        if not hasattr(item, '__iter__'):
            item = [item]
        if not isinstance(item, np.ndarray):
            item = np.array(item)
        return np.concatenate((d, item))

def select(d, index):
    # TODO: support NumPy efficient indexing, pandas, etc.
    return d[index]


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
        if isinstance(index, (int, long)):
            return wrap(self._d, index)
        else:
            return wrap(self._d[index], self._index)
        
    def append(self, d):
        for key, val in iteritems(d):
            self._d[key] = append(self._d[key], val)
        
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
    else:
        if index is None:
            return d
        else:
            return select(d, index)
    