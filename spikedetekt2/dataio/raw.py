"""Reading raw data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
from nose import with_setup

from experiment import Experiment
from chunks import chunk_bounds, Chunk, Excerpt, excerpts
from spikedetekt2.utils.six import Iterator, string_types


# -----------------------------------------------------------------------------
# Raw data readers
# -----------------------------------------------------------------------------
class BaseRawDataReader(object):
    def __init__(self, data, dtype=None):
        self._data = data
        self.dtype = dtype
        self.nsamples, self.nchannels = data.shape
        
    def chunks(self, chunk_size=None, 
                     chunk_overlap=None):
        assert chunk_size is not None, "You need to specify a chunk size."""
        for bounds in chunk_bounds(self._data.shape[0], 
                                   chunk_size=chunk_size, 
                                   overlap=chunk_overlap):
            yield Chunk(self._data, bounds=bounds,)
        
    def excerpts(self, nexcerpts=None, excerpt_size=None):
        for bounds in excerpts(self._data.shape[0],
                               nexcerpts=nexcerpts, 
                               excerpt_size=excerpt_size):
            yield Excerpt(self._data, bounds=bounds)

class NumPyRawDataReader(BaseRawDataReader):
    pass

class DatRawDataReader(BaseRawDataReader):
    """Read a DAT file by chunks."""
    def __init__(self, filenames, dtype=None, shape=None):
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.dtype = np.dtype(dtype)
        self._data = None
        _, self.nchannels = shape
        
    def next_file(self):
        for filename in self.filenames:
            # Find file size.
            size = os.stat(filename).st_size
            row_size = self.nchannels * self.dtype.itemsize
            assert size % row_size == 0
            nsamples = size // row_size
            shape = (nsamples, self.nchannels)
            self._data = np.memmap(filename, dtype=self.dtype,
                                   mode='r',
                                   offset=0,
                                   shape=shape)
            yield filename
            self._data = None
        
    def chunks(self, chunk_size=None, chunk_overlap=None):
        for i, file in enumerate(self.next_file()):
            assert chunk_size is not None, "You need to specify a chunk size."""
            for bounds in chunk_bounds(self._data.shape[0], 
                                       chunk_size=chunk_size, 
                                       overlap=chunk_overlap):
                yield Chunk(self._data, bounds=bounds, dtype=np.float32, 
                            recording=i)
        
    def excerpts(self, nexcerpts=None, excerpt_size=None):
        for i, file in enumerate(self.next_file()):
            for bounds in excerpts(self._data.shape[0],
                                   nexcerpts=nexcerpts, 
                                   excerpt_size=excerpt_size):
                yield Excerpt(self._data, bounds=bounds, dtype=np.float32, 
                              recording=i)
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
        
        
# -----------------------------------------------------------------------------
# Main raw data reading function
# -----------------------------------------------------------------------------
def read_raw(raw, nchannels=None):
    if isinstance(raw, np.ndarray):
        return NumPyRawDataReader(raw)
    elif isinstance(raw, Experiment):
        # TODO: read from Experiment instance
        raise NotImplementedError("Reading from KWIK raw data file (.kwd).")
    elif isinstance(raw, (string_types, list)):
        assert nchannels > 0, ("The number of channels must be specified "
            "in order to read from a .dat file.")
        return DatRawDataReader(raw, dtype=np.int16, shape=(0, nchannels))
    