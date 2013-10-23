"""Files tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.dataio.files import get_filenames, get_basename


# -----------------------------------------------------------------------------
# Files tests
# -----------------------------------------------------------------------------
def test_get_filenames():
    filenames = get_filenames('myexperiment')
    assert os.path.basename(filenames['kwik']) == 'myexperiment.kwik'
    assert os.path.basename(filenames['kwx']) == 'myexperiment.kwx'
    assert os.path.basename(filenames['raw.kwd']) == 'myexperiment.raw.kwd'
    assert os.path.basename(filenames['low.kwd']) == 'myexperiment.low.kwd'
    assert os.path.basename(filenames['high.kwd']) == 'myexperiment.high.kwd'
    
def test_basename_1():
    bn = 'myexperiment'
    filenames = get_filenames(bn)
    kwik = filenames['kwik']
    kwx = filenames['kwx']
    kwdraw = filenames['raw.kwd']
    
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
    
def test_get_file():
    # TODO
    pass
    
    
    