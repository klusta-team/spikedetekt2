"""HDF5 tools tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

import numpy as np
import tables as tb

from spikedetekt2.dataio import create_kwx


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# File creation tests
# -----------------------------------------------------------------------------
def test_create_kwx():
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'myexperiment.kwx')
    
    # Create the KWX file.
    nsamples = 20
    nchannels = 32
    nchannels2 = 24
    nfeatures = 3*nchannels
    channel_groups = {
        0: {},
        1: {'nchannels': nchannels2, 'nfeatures': 3*nchannels2},
        2: {'nfeatures': 2*nchannels},
    }
    
    create_kwx(path, nsamples=nsamples, nchannels=nchannels, 
               nfeatures=nfeatures, channel_groups=channel_groups)
    
    # Open the KWX file.
    f = tb.openFile(path, 'r')
    
    # Group 0
    spiketrain = f.root.channel_groups.channel_group0.spiketrain
    spiketrain.col('sample')
    spiketrain.col('recording')
    spiketrain.col('cluster')
    
    # Group 1
    spikesorting = f.root.channel_groups.channel_group1.spikesorting
    waveforms = f.root.channel_groups.channel_group1.waveforms
    assert spikesorting.col('features').shape[1] == 3*nchannels2
    assert spikesorting.col('masks').shape[1] == 3*nchannels2
    assert waveforms.col('waveform_raw').shape[1] == nsamples*nchannels2

    # Group 2
    spikesorting = f.root.channel_groups.channel_group2.spikesorting
    waveforms = f.root.channel_groups.channel_group2.waveforms
    assert spikesorting.col('features').shape[1] == 2*nchannels
    assert spikesorting.col('masks').shape[1] == 2*nchannels
    assert waveforms.col('waveform_raw').shape[1] == nsamples*nchannels
    
    f.close()
    
    # Delete the file.
    os.remove(path)
    