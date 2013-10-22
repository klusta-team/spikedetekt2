"""HDF5 tools tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

import numpy as np
import tables as tb

from spikedetekt2.dataio.kwik_creation import *


# -----------------------------------------------------------------------------
# HDF5 creation functions tests
# -----------------------------------------------------------------------------
def test_create_kwik():
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'myexperiment.kwik')
    
    prm = {}
    prb = {'channel_groups': [
        {
            'channels': [4, 6, 8],
            'graph': [[4, 6], [8, 4]],
            'geometry': {4: [0.4, 0.6], 6: [0.6, 0.8], 8: [0.8, 0.0]},
        }
    ]}
    
    create_kwik(path, prm=prm, prb=prb)
    
    f = tb.openFile(path, 'r')
    channel = f.root.channel_groups.__getattr__('0').channels.__getattr__('4')
    assert channel._v_attrs.name == 'channel_4'
    
    f.close()
    os.remove(path)
    
def test_create_kwx():
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'myexperiment.kwx')
    
    # Create the KWX file.
    nwavesamples = 20
    nchannels = 32
    nchannels2 = 24
    nfeatures = 3*nchannels
    channel_groups = [
        {},
        {'nchannels': nchannels2, 'nfeatures': 3*nchannels2},
        {'nfeatures': 2*nchannels},
    ]
    
    create_kwx(path, nwavesamples=nwavesamples, nchannels=nchannels, 
               nfeatures=nfeatures, channel_groups=channel_groups)
    
    # Open the KWX file.
    f = tb.openFile(path, 'r')
    
    # Group 1
    fm1 = f.root.channel_groups.__getattr__('1').features_masks
    wr1 = f.root.channel_groups.__getattr__('1').waveforms_raw
    wf1 = f.root.channel_groups.__getattr__('1').waveforms_filtered
    assert fm1.shape[1:] == (3*nchannels2, 2)
    assert wr1.shape[1:] == (nwavesamples, nchannels2)
    assert wf1.shape[1:] == (nwavesamples, nchannels2)

    # Group 2
    fm2 = f.root.channel_groups.__getattr__('2').features_masks
    wr2 = f.root.channel_groups.__getattr__('2').waveforms_raw
    wf2 = f.root.channel_groups.__getattr__('2').waveforms_filtered
    assert fm2.shape[1:] == (2*nchannels, 2)
    assert wr2.shape[1:] == (nwavesamples, nchannels)
    assert wf2.shape[1:] == (nwavesamples, nchannels)
    
    f.close()
    
    # Delete the file.
    os.remove(path)
    
def test_create_kwd():
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'myexperiment.raw.kwd')
    
    # Create the KWD file.
    nchannels_tot = 32*3
    recordings = [
        {'nsamples': 100},
        {},
        {'nsamples': 150},
    ]
    
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
    