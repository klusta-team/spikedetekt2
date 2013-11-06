"""Main module tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio import BaseRawDataReader, read_raw
from spikedetekt2.core import run


# -----------------------------------------------------------------------------
# Processing tests
# -----------------------------------------------------------------------------
def test_run_1():
    sample_rate = 20000.
    duration = 1.
    nsamples = int(sample_rate * duration)
    nfeatures = 3
    nwavesamples = 10
    nchannels = 2
    
    prm = {
        'nfeatures': nfeatures, 
        'nwavesamples': nwavesamples, 
        'nchannels': nchannels,
        'sample_rate': sample_rate,
        
    }
    prb = {'channel_groups': [
        {
            'channels': [0, 1],
            'graph': [[0, 1]],
        }
    ]}
    
    raw = np.random.randn(nsamples, nchannels)
    
    run(raw, prm=prm, prb=prb)
    
    
    
    


