"""Raw data reader tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile
import shutil

import numpy as np
from nose import with_setup

from spikedetekt2.dataio import NumPyRawDataReader, DatRawDataReader


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()
FILENAME = 'mydatfile.dat'
PATH = os.path.join(DIRPATH, FILENAME)
NSAMPLES = 20000
NCHANNELS = 32

def create_trace(nsamples, nchannels):
    noise = np.array(np.random.randint(size=(nsamples, nchannels),
        low=-1000, high=1000), dtype=np.int16)
    t = np.linspace(0., 100., nsamples)
    low = np.array(10000 * np.cos(t), dtype=np.int16)
    return noise + low[:, np.newaxis]
    
def dat_setup_1():
    trace = create_trace(NSAMPLES, NCHANNELS)
    trace.tofile(PATH)
    
def dat_teardown_1():
    os.remove(PATH)
    
    
# -----------------------------------------------------------------------------
# Chunking Tests
# -----------------------------------------------------------------------------
def test_raw_data_1():
    data = np.random.randn(200, 4)
    rd = NumPyRawDataReader(data)
    chunks = rd.chunks(chunk_size=100, chunk_overlap=20)
    
    ch = next(chunks)
    assert ch.window_full == (0, 100)
    assert ch.window_keep == (0, 90)
    assert np.array_equal(ch.data_chunk_full, data[0:100])
    assert np.array_equal(ch.data_chunk_keep, data[0:90])
    
    ch = next(chunks)
    assert ch.window_full == (80, 180)
    assert ch.window_keep == (90, 170)
    assert np.array_equal(ch.data_chunk_full, data[80:180])
    assert np.array_equal(ch.data_chunk_keep, data[90:170])
    
    assert str(ch)
    
@with_setup(dat_setup_1, dat_teardown_1)
def test_raw_dat_1():
    with DatRawDataReader(PATH, dtype=np.int16, shape=(-1, NCHANNELS)) as reader:
        assert len([chunk for chunk in reader.chunks(NSAMPLES, 0)]) == 1
    
    
def test_raw_data_iterator():
    data = np.random.randn(200, 4)
    rd = NumPyRawDataReader(data)
    assert len([ch for ch in rd.chunks(chunk_size=100, chunk_overlap=20)]) == 3
    
    