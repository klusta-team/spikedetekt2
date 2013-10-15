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
def _select(obj, indices):
    for index in indices:
        try:
            obj = obj[index]
            indices = indices[1:]
            break
        except:
            pass
    return obj, indices

def get_node(obj, path):
    """Retrieve a deep object based on a path. Return either a Wrapped instance if the deep object is not a node, or another type of object."""
    subobj = obj
    indices = []
    for item in path:
        try:
            subobj = subobj[item]
        except Exception as e:
            # print e
            indices.append(item)
        subobj, indices = _select(subobj, indices)
    if isinstance(subobj, dict) or (isinstance(subobj, list) and 
                                    subobj and 
                                    isinstance(subobj[0], dict)):
        return Wrapped(obj, path)
    else:
        assert not indices, "This path does not exist."
        return subobj

class Wrapped(object):
    def __init__(self, obj, path=[]):
        self._obj = obj
        self._path = path
        
    def _down(self, key):
        return get_node(self._obj, self._path + [key])
        
    def __getattr__(self, key):
        return self._down(key)
    
    def __getitem__(self, key):
        return self._down(key)
    
    def __repr__(self):
        return "Wrapped " + str(self._obj) + "  :  " + str(self._path)

def wrap(obj):
    return Wrapped(obj)
    