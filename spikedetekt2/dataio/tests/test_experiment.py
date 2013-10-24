"""Experiment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import tables as tb

from spikedetekt2.dataio import (add_recording, create_files, open_files,
    close_files, add_event_type, add_cluster_group, get_filenames)
from spikedetekt2.utils.six import itervalues


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()

def setup():
    # Create files.
    prm = {}
    prb = {'channel_groups': [
        {
            'channels': [4, 6, 8],
            'graph': [[4, 6], [8, 4]],
            'geometry': {4: [0.4, 0.6], 6: [0.6, 0.8], 8: [0.8, 0.0]},
        }
    ]}
    create_files('myexperiment', dir=DIRPATH, prm=prm, prb=prb)
    
    # Open the files.
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    
    # Add data.
    add_recording(files, 
                  sample_rate=20000.,
                  start_time=10., 
                  start_sample=200000.,
                  bit_depth=16,
                  band_high=100.,
                  band_low=500.)
    add_event_type(files, 'myevents')
    add_cluster_group(files, channel_group_id='0', id='noise', name='Noise')
    
    # Close the files
    close_files(files)

def teardown():
    files = get_filenames('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]


# -----------------------------------------------------------------------------
# Experiment creation tests
# -----------------------------------------------------------------------------
def test_experiment_1():
    pass


