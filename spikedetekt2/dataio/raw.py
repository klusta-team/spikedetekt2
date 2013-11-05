"""Reading raw data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from chunks import chunk_bounds, Chunk


# -----------------------------------------------------------------------------
# Raw data readers
# -----------------------------------------------------------------------------
class NumPyRawDataReader(object):
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
                                    
        