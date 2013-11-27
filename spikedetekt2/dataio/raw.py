"""Reading raw data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from chunks import chunk_bounds, Chunk, Excerpt, excerpts
from six import Iterator


# -----------------------------------------------------------------------------
# Raw data readers
# -----------------------------------------------------------------------------
class BaseRawDataReader(object):
    def __init__(self, data, dtype=None, recording=0):
        self._data = data
        self.dtype = dtype
        self.recording = recording
        self.nsamples, self.nchannels = data.shape
        
    def chunks(self, chunk_size=None, 
                     chunk_overlap=None):
        assert chunk_size is not None, "You need to specify a chunk size."""
        for bounds in chunk_bounds(self._data.shape[0], 
                                   chunk_size=chunk_size, 
                                   overlap=chunk_overlap):
            yield Chunk(self._data, bounds=bounds, dtype=self.dtype,
                        recording=self.recording)
        
    def excerpts(self, nexcerpts=None, excerpt_size=None):
        for bounds in excerpts(self._data.shape[0],
                               nexcerpts=nexcerpts, 
                               excerpt_size=excerpt_size):
            yield Excerpt(self._data, bounds=bounds)

class NumPyRawDataReader(BaseRawDataReader):
    pass

class DatRawDataReader(BaseRawDataReader):
    """Read a DAT file by chunks."""
    # TODO
    pass
    
def read_raw(raw):
    if isinstance(raw, np.ndarray):
        return NumPyRawDataReader(raw)
    elif isinstance(raw, Experiment):
        # TODO: read from Experiment instance
        pass
    elif isinstance(raw, string_types):
        # TODO: read from .dat file
        pass
    