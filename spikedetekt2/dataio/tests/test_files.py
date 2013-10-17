"""Files tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.dataio.files import generate_filenames


# -----------------------------------------------------------------------------
# Files tests
# -----------------------------------------------------------------------------
def test_generate_filenames():
    filenames = generate_filenames('myexperiment')
    assert filenames['kwik'] == 'myexperiment.kwik'
    assert filenames['kwx'] == 'myexperiment.kwx'
    assert filenames['kwd']['raw'] == 'myexperiment.raw.kwd'
    assert filenames['kwd']['low'] == 'myexperiment.low.kwd'
    assert filenames['kwd']['high'] == 'myexperiment.high.kwd'
    assert filenames['kwe'] == 'myexperiment.kwe'
    
    