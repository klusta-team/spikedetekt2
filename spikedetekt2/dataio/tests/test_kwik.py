"""HDF5 tools tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

import numpy as np
import tables as tb

from spikedetekt2.dataio import create_kwx, create_kwd


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
    nwavesamples = 20
    nchannels = 32
    nchannels2 = 24
    nfeatures = 3*nchannels
    channel_groups = {
        0: {},
        1: {'nchannels': nchannels2, 'nfeatures': 3*nchannels2},
        2: {'nfeatures': 2*nchannels},
    }
    
    create_kwx(path, nwavesamples=nwavesamples, nchannels=nchannels, 
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
    assert waveforms.col('waveform_raw').shape[1] == nwavesamples*nchannels2

    # Group 2
    spikesorting = f.root.channel_groups.channel_group2.spikesorting
    waveforms = f.root.channel_groups.channel_group2.waveforms
    assert spikesorting.col('features').shape[1] == 2*nchannels
    assert spikesorting.col('masks').shape[1] == 2*nchannels
    assert waveforms.col('waveform_raw').shape[1] == nwavesamples*nchannels
    
    f.close()
    
    # Delete the file.
    os.remove(path)
    
def test_create_kwd():
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'myexperiment.raw.kwd')
    
    # Create the KWD file.
    nchannels_tot = 32*3
    recordings = {
        0: {'nsamples': 100},
        1: {},
        2: {'nsamples': 150},
    }
    
    create_kwd(path, type='raw', nchannels_tot=nchannels_tot, 
               recordings=recordings,)
    
    # Open the KWX file.
    f = tb.openFile(path, 'r')
    
    assert f.root.recording0.data_raw.shape[1] == nchannels_tot
    assert f.root.recording1.data_raw.shape[1] == nchannels_tot
    assert f.root.recording2.data_raw.shape[1] == nchannels_tot
    
    f.close()
    
    # Delete the file.
    os.remove(path)
    