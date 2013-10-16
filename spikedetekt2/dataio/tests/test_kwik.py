"""HDF5 tools tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import tempfile

import numpy as np
import tables as tb

from spikedetekt2.dataio.kwik import *


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# KWIK file creation tests
# -----------------------------------------------------------------------------
def test_create_kwik():
    kwik = create_kwik(
        name='my experiment',
        channel_groups=[
            create_kwik_channel_group(
                ichannel_group=0,
                name='my channel group',
                graph=[[0,1], [1, 2]],
                channels=[
                    create_kwik_channel(
                        name='my first channel',
                        ignored=False,
                        position=[0., 0.],
                        voltage_gain=10.,
                        display_threshold=None),
                    create_kwik_channel(
                        name='my second channel',
                        ignored=True,
                        position=[1., 1.],
                        voltage_gain=20.,
                        display_threshold=None),
                    create_kwik_channel(
                        name='my third channel',
                        ignored=False,
                        position=[0., 2.],
                        voltage_gain=30.,
                        display_threshold=None),
                ],
                cluster_groups=[
                    create_kwik_cluster_group(color=2, name='my cluster group',
                        clusters=[
                             create_kwik_cluster(color=4),
                        ])
                ],
            )
        ],
        recordings=[
            create_kwik_recording(
                irecording=0,
                start_time=0.,
                start_sample=0,
                sample_rate=20000.,
                band_low=100.,
                band_high=3000.,
                bit_depth=16),
        ],
        event_types=[
            create_kwik_event_type(
                color=3,
            ),
        ],
    )
    assert kwik['VERSION'] == 2
    assert kwik['name'] == 'my experiment'
    
    channel_group = kwik['channel_groups'][0]
    assert channel_group['name'] == 'my channel group'
    assert channel_group['graph'] == [[0, 1], [1, 2]]
    
    channels = channel_group['channels']
    assert channels[0]['name'] == 'my first channel'
    assert channels[1]['name'] == 'my second channel'
    assert channels[2]['name'] == 'my third channel'
    
    assert not channels[0]['ignored']
    assert channels[1]['ignored']
    assert not channels[2]['ignored']
    
    cluster_group = channel_group['cluster_groups'][0]
    cluster = cluster_group['clusters'][0]
    assert cluster['application_data']['klustaviewa']['color'] == 4
    
    recording = kwik['recordings'][0]
    assert recording['sample_rate'] == 20000.
    
    event_type = kwik['event_types'][0]
    assert event_type['application_data']['klustaviewa']['color'] == 3
    

# -----------------------------------------------------------------------------
# HDF5 helper functions tests
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
    spiketrain.col('fractional')
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
    
def test_create_kwe():
    dirpath = tempfile.mkdtemp()
    path = os.path.join(dirpath, 'myexperiment.kwe')
    
    create_kwe(path)
    
    # Open the KWX file.
    f = tb.openFile(path, 'r')
    
    # Group 0
    events = f.root.events
    events.col('sample')
    events.col('recording')
    events.col('event_type')
    
    event_types = f.root.event_types
    event_types.col('name')
    
    f.close()
    
    # Delete the file.
    os.remove(path)
  