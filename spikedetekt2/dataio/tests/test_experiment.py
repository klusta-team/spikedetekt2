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

from spikedetekt2.dataio.kwik import (add_recording, create_files, open_files,
    close_files, add_event_type, add_cluster_group, get_filenames,
    add_cluster)
from spikedetekt2.dataio.experiment import (Experiment, _resolve_hdf5_path,)
from spikedetekt2.utils.six import itervalues


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()

def setup():
    # Create files.
    prm = {'nfeatures': 3, 'nwavesamples': 10, 'nchannels': 10}
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
                  band_low=500.,
                  nchannels=10,)
    add_event_type(files, 'myevents')
    add_cluster_group(files, channel_group_id='0', id='noise', name='Noise')
    add_cluster(files, channel_group_id='0',)
    
    # Close the files
    close_files(files)

def teardown():
    files = get_filenames('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]


# -----------------------------------------------------------------------------
# Experiment creation tests
# -----------------------------------------------------------------------------
def test_resolve_hdf5_path():
    path = "{kwx}/channel_groups/0"
    
    files = open_files('myexperiment', dir=DIRPATH)
    assert _resolve_hdf5_path(files, path)
    
    close_files(files)

def test_experiment_1():
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        assert exp.name == 'myexperiment'
        assert exp.application_data
        assert exp.user_data
        
        # Channel group.
        chgrp = exp.channel_groups[0]
        assert chgrp.name == 'channel_group_0'
        assert chgrp.adjacency_graph == [[4, 6], [8, 4]]
        assert chgrp.application_data
        assert chgrp.user_data
        
        # Channels.
        channels = chgrp.channels
        assert list(sorted(channels.keys())) == [4, 6, 8]
        
        # Channel.
        ch = channels[4]
        assert ch.name == 'channel_4'
        ch.kwd_index 
        ch.ignored 
        assert ch.position == [.4, .6]
        ch.voltage_gain 
        ch.display_threshold 
        assert ch.application_data
        assert ch.user_data
        
        # Spikes.
        spikes = chgrp.spikes
        assert isinstance(spikes.time_samples, tb.EArray)
        assert spikes.time_samples.dtype == np.uint64
        assert spikes.time_samples.ndim == 1
        
        assert isinstance(spikes.time_fractional, tb.EArray)
        assert spikes.time_fractional.dtype == np.uint8
        assert spikes.time_fractional.ndim == 1
        
        assert isinstance(spikes.recording, tb.EArray)
        assert spikes.recording.dtype == np.uint16
        assert spikes.recording.ndim == 1
        
        assert isinstance(spikes.cluster, tb.EArray)
        assert spikes.cluster.dtype == np.uint32
        assert spikes.cluster.ndim == 1
        
        assert isinstance(spikes.cluster_original, tb.EArray)
        assert spikes.cluster_original.dtype == np.uint32
        assert spikes.cluster_original.ndim == 1
        
        assert isinstance(spikes.features_masks, tb.EArray)
        assert spikes.features_masks.dtype == np.float32
        assert spikes.features_masks.ndim == 3
        
        assert isinstance(spikes.waveforms_raw, tb.EArray)
        assert spikes.waveforms_raw.dtype == np.int16
        assert spikes.waveforms_raw.ndim == 3
        
        assert isinstance(spikes.waveforms_filtered, tb.EArray)
        assert spikes.waveforms_filtered.dtype == np.int16
        assert spikes.waveforms_filtered.ndim == 3
        
        # Cluster.
        cluster = chgrp.clusters[0]
        
        assert cluster.application_data
        assert cluster.user_data
        assert cluster.quality_measures
        
        cluster.cluster_group
        cluster.mean_waveform_raw
        cluster.mean_waveform_filtered
        
        # Cluster group.
        cluster_group = chgrp.cluster_groups['noise']
        assert cluster_group.name == 'Noise'
        
        assert cluster_group.application_data
        assert cluster_group.user_data
        
        # Recordings.
        rec = exp.recordings[0]
        assert rec.name == 'recording_0'
        assert rec.sample_rate == 20000.
        assert rec.start_time == 10.
        assert rec.start_sample == 200000.
        assert rec.bit_depth == 16
        assert rec.band_high == 100.
        assert rec.band_low == 500.
        
        rd = rec.raw.data
        assert isinstance(rd, tb.EArray)
        assert rd.shape == (0, 10)
        assert rd.dtype == np.int16
        
        # Event types
        evtp = exp.event_types['myevents']
        evtp.application_data
        evtp.user_data
        
        samples = evtp.events.time_samples
        assert isinstance(samples, tb.EArray)
        assert samples.dtype == np.uint64
        
        recordings = evtp.events.recording
        assert isinstance(recordings, tb.EArray)
        assert recordings.dtype == np.uint16
        
        evtp.events.user_data