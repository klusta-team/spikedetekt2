"""Reading raw data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from chunks import chunk_bounds, Chunk
from spikedetekt2 import six


# -----------------------------------------------------------------------------
# Raw data readers
# -----------------------------------------------------------------------------
class BaseRawDataReader(object):
    pass

class NumPyRawDataReader(BaseRawDataReader):
    """Read a NumPy array with raw data by chunks."""
    def __init__(self, arr, chunk_size=None, chunk_overlap=None):
        self._arr = arr
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nsamples, self.nchannels = arr.shape
        self.reset()

    def next_chunk(self):
        return Chunk(self._arr, bounds=next(self._chunks))
        
    def reset(self):
        self._chunks = chunk_bounds(self.nsamples, self.chunk_size, 
                                    overlap=self.chunk_overlap)

class DatRawDataReader(BaseRawDataReader):
    """Read a DAT file by chunks."""
    # TODO
    pass
    
def read_raw(raw, **kwargs):
    if isinstance(raw, np.ndarray):
        return NumPyRawDataReader(raw, **kwargs)
    elif isinstance(raw, string_types):
        # TODO
        pass
    