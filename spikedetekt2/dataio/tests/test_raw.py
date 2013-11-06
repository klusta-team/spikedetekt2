"""Raw data reader tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio import NumPyRawDataReader


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
    
def test_raw_data_iterator():
    data = np.random.randn(200, 4)
    rd = NumPyRawDataReader(data)
    assert len([ch for ch in rd.chunks(chunk_size=100, chunk_overlap=20)]) == 3
    
    