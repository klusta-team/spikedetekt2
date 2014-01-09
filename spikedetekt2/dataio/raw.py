"""Reading raw data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import tables as tb
from nose import with_setup

from kwik import open_file, add_recording_in_kwd
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
    def __init__(self, filenames, dtype=None, dtype_to=None, 
                 shape=None, len=None):
        """
        
        Arguments:
        * dtype: the dtype of the DAT file.
        * dtype_to: the dtype of the chunks to read.
        * len: length of the array. By default, None = full length of the 
          data.
        """
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        
        if dtype is None:
            dtype = np.int16
        self.dtype = np.dtype(dtype)
        
        if dtype_to is None:
            dtype_to = np.int16
        self.dtype_to = np.dtype(dtype_to)
        
        self.len = len
        self._data = None
        _, self.nchannels = shape
        
    def next_file(self):
        for filename in self.filenames:
            # Find file size.
            size = os.stat(filename).st_size
            row_size = self.nchannels * self.dtype.itemsize
            assert size % row_size == 0
            self.nsamples = size // row_size
            shape = (self.nsamples, self.nchannels)
            self._data = np.memmap(filename, dtype=self.dtype,
                                   mode='r',
                                   offset=0,
                                   shape=shape)
            yield filename
            self._data = None
        
    def chunks(self, chunk_size=None, chunk_overlap=0):
        for i, file in enumerate(self.next_file()):
            assert chunk_size is not None, "You need to specify a chunk size."""
            # Use the full length of the data...
            if self.len is None:
                len = self._data.shape[0]
            else:
            # ... or restrict the length of the file.
                len = self.len
            for bounds in chunk_bounds(len, 
                                       chunk_size=chunk_size, 
                                       overlap=chunk_overlap):
                yield Chunk(self._data, bounds=bounds, dtype=self.dtype_to, 
                            recording=i)
        
    def excerpts(self, nexcerpts=None, excerpt_size=None):
        for i, file in enumerate(self.next_file()):
            for bounds in excerpts(self._data.shape[0],
                                   nexcerpts=nexcerpts, 
                                   excerpt_size=excerpt_size):
                yield Excerpt(self._data, bounds=bounds, dtype=self.dtype_to, 
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
        if raw.endswith('.dat'):
            assert nchannels > 0, ("The number of channels must be specified "
                "in order to read from a .dat file.")
            return DatRawDataReader(raw, dtype=np.int16, shape=(0, nchannels))
        elif raw.endswith('.kwd'):
            raise NotImplementedError(("Reading raw data from KWD files is not"
            " implemented yet."))
        else:
            raise ArgumentError("Unknown file extension for the raw data.")
            
def convert_dat_to_kwd(dat_reader, kwd_file):
    with open_file(kwd_file, 'a') as kwd:
        for chunk in dat_reader.chunks(20000):
            data = chunk.data_chunk_full
            rec = chunk.recording
            try:
                # Add the data to the KWD file, in /recordings/[X]/data.
                kwd.root.recordings._f_getChild(str(rec)).data.append(data)
            except tb.NoSuchNodeError:
                # If /recordings/[X] does not exist, add this recording
                # to the KWD file and the data as well.
                add_recording_in_kwd(kwd, recording_id=rec,
                                     nchannels=chunk.nchannels,
                                     nsamples=chunk.nsamples,
                                     data=data)
                