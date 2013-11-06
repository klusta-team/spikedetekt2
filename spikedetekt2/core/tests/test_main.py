"""Main module tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

import numpy as np

from spikedetekt2.dataio import (BaseRawDataReader, read_raw, create_files,
    open_files, close_files, add_recording, add_cluster_group, add_cluster,
    get_filenames, Experiment, excerpts)
from spikedetekt2.core import run
from spikedetekt2.utils import itervalues, get_params


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()

sample_rate = 20000
duration = 10.
nfeatures = 3
nwavesamples = 10
nchannels = 2
chunk_size = 20000
nsamples = int(sample_rate * duration)
raw_data = .1 * np.random.randn(nsamples, nchannels)

prm = get_params(**{
    'nfeatures': nfeatures, 
    'nwavesamples': nwavesamples, 
    'nchannels': nchannels,
    'sample_rate': sample_rate,
    'chunk_size': chunk_size,
})
prb = {'channel_groups': [
    {
        'channels': [0, 1],
        'graph': [[0, 1]],
    }
]}

def setup():
    create_files('myexperiment', dir=DIRPATH, prm=prm, prb=prb)
    
    # Open the files.
    files = open_files('myexperiment', dir=DIRPATH, mode='a')
    
    # Add data.
    add_recording(files, 
                  sample_rate=sample_rate,
                  nchannels=nchannels)
    add_cluster_group(files, channel_group_id='0', id='noise', name='Noise')
    add_cluster(files, channel_group_id='0',)
    
    # Close the files
    close_files(files)

def teardown():
    files = get_filenames('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]


# -----------------------------------------------------------------------------
# Processing tests
# -----------------------------------------------------------------------------
def test_run_1():
    
    # Add "spikes".
    for start, end in excerpts(nsamples, nexcerpts=100, excerpt_size=10):
        raw_data[start:end] *= 5
    
    # Run the algorithm.
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        run(raw_data, experiment=exp, prm=prm, prb=prb)
    
    # Open the data files.
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        print(len(exp.channel_groups[0].spikes))
    
    


