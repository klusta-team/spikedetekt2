"""Wrap dictionaries in Python objects for easy access of hierarchical structures."""

# -----------------------------------------------------------------------------
# Wrap functions
# -----------------------------------------------------------------------------
class WrappedIndexed(object):
    def __init__(self, d, index):
        self._d = d
        self._index = index
        
    def __getattr__(self, key):
        val = self._d[key]
        if isinstance(val, dict):
            return WrappedIndexed(val, self._index)
        else:
            return val[self._index]

class Wrapped(object):
    def __init__(self, d):
        self._d = d
        
    def __getattr__(self, key):
        val = self._d[key]
        return wrap(val)
            
    def __getitem__(self, index):
        return WrappedIndexed(self._d, index)
        
    def __dir__(self):
        return self._d.__dir__()

    def __repr__(self):
        return self._d.__repr__()
        
def wrap(d):
    if isinstance(d, dict):
        return Wrapped(d)
    elif isinstance(d, list):
        return [wrap(k) for k in d]
    else:
        return d
    