"""Files tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.dataio.files import generate_filenames, get_basename


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
    
def test_basename_1():
    bn = 'myexperiment'
    filenames = generate_filenames(bn)
    kwik = filenames['kwik']
    kwx = filenames['kwx']
    kwdraw = filenames['kwd']['raw']
    
    assert get_basename(kwik) == bn
    assert get_basename(kwx) == bn
    assert get_basename(kwdraw) == bn
    
def test_basename_2():
    kwik = '/my/path/experiment.kwik'
    kwx = '/my/path/experiment.kwx'
    kwdhigh = '/my/path/experiment.high.kwd'
    
    assert get_basename(kwik) == 'experiment'
    assert get_basename(kwx) == 'experiment'
    assert get_basename(kwdhigh) == 'experiment'
    