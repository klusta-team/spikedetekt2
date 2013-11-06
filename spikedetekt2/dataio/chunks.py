"""Chunking routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# Chunking functions
# -----------------------------------------------------------------------------
def chunk_bounds(n_samples, chunk_size, overlap=0):
    """Returns chunks of the form:
    [ overlap/2 | chunk_size-overlap | overlap/2 ]
    s_start   keep_start           keep_end     s_end
    Except for the first and last chunks which do not have a left/right overlap
    
    """
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end-overlap//2
    yield s_start,s_end,keep_start,keep_end
    
    while s_end-overlap+chunk_size < n_samples:
        s_start = s_end-overlap
        s_end = s_start+chunk_size
        keep_start = keep_end
        keep_end = s_end-overlap//2
        yield s_start,s_end,keep_start,keep_end
        
    s_start = s_end-overlap
    s_end = n_samples
    keep_start = keep_end
    keep_end = s_end
    yield s_start,s_end,keep_start,keep_end


# -----------------------------------------------------------------------------
# Chunk class
# -----------------------------------------------------------------------------
class Chunk(object):
    def __init__(self, data=None, nsamples=None, nchannels=None,
                 bounds=None):
        self._data = data
        if nsamples is None and nchannels is None:
            nsamples, nchannels = data.shape
        self.nsamples = nsamples
        self.nchannels = nchannels
        self._s_start, self._s_end, self._keep_start, self._keep_end = bounds
        self.window_full = self._s_start, self._s_end
        self.window_keep = self._keep_start, self._keep_end
        
    @property
    def data_chunk_full(self):
        return self._data[self._s_start:self._s_end,:]
    
    @property
    def data_chunk_keep(self):
        return self._data[self._keep_start:self._keep_end,:]
    
    def __repr__(self):
        return "<Chunk [{0:d}|{1:d}|{2:d}|{3:d}], maxlen={4:d}>".format(
            self._s_start, self._keep_start, self._keep_end, self._s_end,
            self._data.shape[0]
        )
        
        