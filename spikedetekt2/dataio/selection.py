"""Functions for selecting portions of arrays."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np
import pandas as pd
import tables as tb


# -----------------------------------------------------------------------------
# Selection functions
# -----------------------------------------------------------------------------
def select_numpy(data, indices):
    """Select a portion of an array with the corresponding indices.
    The first axis of data corresponds to the indices."""
    
    if not hasattr(indices, '__len__'):
        return data[indices, ...]
    
    if type(indices) == list:
        indices = np.array(indices)
    
    assert isinstance(data, np.ndarray)
    assert isinstance(indices, np.ndarray)
    
    # indices can contain boolean masks...
    if indices.dtype == np.bool:
        data_selection = np.compress(indices, data, axis=0)
    # or indices.
    else:
        data_selection = np.take(data, indices, axis=0)
    return data_selection

def select_pandas(data, indices, drop_empty_rows=True):
    if isinstance(indices, slice):
        return np.array(data.iloc[indices]).squeeze()
    elif not hasattr(indices, '__len__'):
        try:
            return np.array(data.ix[indices]).squeeze()
        except KeyError:
            raise IndexError("Index {0:d} is not in the data.".format(
                indices))
    
    try:
        # Remove empty rows.
        data_selected = data.ix[indices]
    except IndexError:
        # This exception happens if the data is a view of the whole array,
        # and `indices` is an array of booleans adapted to the whole array and 
        # not to the view. So we convert `indices` into an array of indices,
        # so that Pandas can handle missing values.
        data_selected = data.ix[np.nonzero(indices)[0]]
    if drop_empty_rows:
        data_selected = data_selected.dropna()
    return data_selected
    
def slice_to_indices(indices, stop=None, lenindices=None):
    start, step = (indices.start or 0), (indices.step or 1)
    if not stop:
        assert lenindices is not None
        # Infer stop such that indices and values have the same size.
        stop = np.floor(start + step*lenindices)
    indices = np.arange(start, stop, step)
    if not lenindices:
        lenindices = len(indices)
    assert len(indices) == lenindices
    return indices

def pandaize(values, indices):
    """Convert a NumPy array to a Pandas object, with the indices indices.
    values contains already selected data."""
    if isinstance(indices, (int, long)):
        indices = [indices]
    if isinstance(indices, list):
        indices = np.array(indices)
    # Get the indices.
    if isinstance(indices, slice):
        indices = slice_to_indices(indices, lenindices=len(values))
    elif indices.dtype == np.bool:
        indices = np.nonzero(indices)[0]
    
    # Create the Pandas object with the indices.
    if values.ndim == 1:
        pd_arr = pd.Series(values, index=indices)
    elif values.ndim == 2:
        pd_arr = pd.DataFrame(values, index=indices)
    elif values.ndim == 3:
        pd_arr = pd.Panel(values, items=indices)
    return pd_arr
    
def select_pytables(data, indices, process_fun=None):
    values = data[indices,...]
    # Process the NumPy array.
    if process_fun:
        values = process_fun(values)
    return pandaize(values, indices)
    
def select(data, indices=None):
    """Select portion of the data, with the only assumption that indices are
    along the first axis.
    
    data can be a NumPy or Pandas object.
    
    """
    # indices=None or 'all' means select all.
    if indices is None or indices is 'all':
        if type(data) == tuple:
            indices = np.ones(data[0].shape[0], dtype=np.bool)
        else:
            return data
        
    if not hasattr(indices, '__len__') and not isinstance(indices, slice):
        indices = [indices]
    indices_argument = indices
        
    # Ensure indices is an array of indices or boolean masks.
    if not isinstance(indices, np.ndarray) and not isinstance(indices, slice):
        # Deal with empty indices.
        if not len(indices):
            if data.ndim == 1:
                return np.array([])
            elif data.ndim == 2:
                return np.array([[]])
            elif data.ndim == 3:
                return np.array([[[]]])
        else:
            if type(indices[0]) in (int, np.int32, np.int64):
                indices = np.array(indices, dtype=np.int32)
            elif type(indices[0]) == bool:
                indices = np.array(indices, dtype=np.bool)
            else:
                indices = np.array(indices)
    
    # Use NumPy, PyTables (tuple) or Pandas version
    if type(data) == np.ndarray:
        if data.size == 0:
            return data
        return select_numpy(data, indices_argument)
    elif type(data) in (tuple, tb.EArray):
        if type(data) == tuple:
            data, process_fun = data
        else:
            process_fun = None
        return select_pytables(data, indices_argument,
                               process_fun=process_fun)
    elif hasattr(data, 'values'):
        if data.values.size == 0:
            return data
        return select_pandas(data, indices_argument)
    else:
        return select_pytables(data, indices_argument)