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

from spikedetekt2.dataio.experiment import create_experiment, HDF5Proxy
from spikedetekt2.dataio.tests.mock import mock_experiment


# -----------------------------------------------------------------------------
# Experiment creation tests
# -----------------------------------------------------------------------------
def test_create_experiment():
    """Create an experiment (write some files)."""
    dir = tempfile.mkdtemp()
    name = 'my experiment'
    mock_experiment(dir=dir, name=name)
    shutil.rmtree(dir)
    
    
# -----------------------------------------------------------------------------
# Experiment class tests
# -----------------------------------------------------------------------------
# def test_experiment_spikes():
    # spikes = Spikes()
    # spikes.features[::2]
    # spikes.features[::2]

def test_experiment_class():
    dir = tempfile.mkdtemp()
    name = 'myexperiment'
    mock_experiment(dir=dir, name=name)
    
    # Modify the experiment files.
    path = os.path.join(dir, name + '.kwx')
    f = tb.openFile(path, 'r+')
    
    # Add some spikes.
    spiketrain = f.root.channel_groups.channel_group0.spiketrain
    spikesorting = f.root.channel_groups.channel_group0.spikesorting
    for i in range(10):
        row = spiketrain.row
        row['sample'] = i*1000
        row['fractional'] = 0.
        row['recording'] = 0
        row['cluster'] = i % 2
        row.append()
    
        row = spikesorting.row
        row['features'] = np.random.randint(size=6, low=-100, high=100)
        row['masks'] = np.random.randint(size=6, low=0, high=255)
        row['cluster_original'] = i % 2
        row.append()
        
    f.flush()
    f.close()
    
    # Now, read the file.
    f = tb.openFile(path, 'r')
    spiketrain = f.root.channel_groups.channel_group0.spiketrain
    spikesorting = f.root.channel_groups.channel_group0.spikesorting
    
    spikes = HDF5Proxy(
        sample=spiketrain,
        fractional=spiketrain,
        recording=spiketrain,
        cluster=spiketrain,
        
        features=spikesorting,
        masks=spikesorting,
        cluster_original=spikesorting,
    )
    
    for clusters in (spikes.cluster[:], spikes[:].cluster):
        assert isinstance(clusters, pd.Series)
        assert np.array_equal(clusters.index, np.arange(10))
        assert np.array_equal(clusters.values, [0, 1] * 5)
    
    for clusters in (spikes.cluster[1::2], spikes[1::2].cluster):
        assert isinstance(clusters, pd.Series)
        assert np.array_equal(clusters.index, np.arange(1, 10, 2))
        assert np.array_equal(clusters.values, np.ones(5))
    
    f.close()
    shutil.rmtree(dir)

