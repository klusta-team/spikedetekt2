"""Reading raw data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from chunks import chunk_bounds, Chunk
from six import Iterator


# -----------------------------------------------------------------------------
# Raw data readers
# -----------------------------------------------------------------------------
class Excerpts(Iterator):
    def __init__(self, data=None, nexcerpts=None, excerpt_size=None):
        self.data = data
        self.nsamples = data.shape[0]
        self.nexcerpts = nexcerpts
        self.excerpt_size = excerpt_size
        self.step = (self.nsamples - self.excerpt_size) // self.nexcerpts
    
    def __next__(self):
        for i in range(self.nexcerpts):
            start = i * self.step
            end = start + self.excerpt_size
        return Excerpt(self.data, start=start, end=end)
        
    def __iter__(self):
        return self

class BaseRawDataReader(Iterator):
    def __next__(self):
        return self.next_chunk()
        
    def next_chunk(self):
        return
        
    def excerpts(self, nexercepts=None, excerpt_size=None):
        return Excerpts(self._data,
                        nexercepts=nexercepts, excerpt_size=excerpt_size)
        
    def reset(self):
        return
        
    def __iter__(self):
        return self

class NumPyRawDataReader(BaseRawDataReader):
    """Read a NumPy array with raw data by chunks."""
    def __init__(self, data, chunk_size=None, chunk_overlap=0):
        assert chunk_size is not None, "You need to specify a chunk size."""
        self._data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nsamples, self.nchannels = data.shape
        self.reset()

    def next_chunk(self):
        return Chunk(self._data, bounds=next(self._chunks))
        
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
    elif isinstance(raw, Experiment):
        # TODO: read from Experiment instance
        pass
    elif isinstance(raw, string_types):
        # TODO: read from .dat file
        pass
    