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

from spikedetekt2.dataio.hdf5proxy import HDF5Proxy
from spikedetekt2.dataio.tests.mock import mock_experiment


# -----------------------------------------------------------------------------
# Experiment class tests
# -----------------------------------------------------------------------------
def test_hdf5_proxy():
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
    
    for fractionals in (spikes.fractional[1:-2:3], spikes[1:-2:3].fractional):
        assert isinstance(fractionals, pd.Series)
        assert np.array_equal(fractionals.index, [1, 4, 7])
        assert np.array_equal(fractionals.values, [0., 0., 0.])
    
    features = spikes.features[:]
    assert isinstance(features, pd.DataFrame)
    assert features.values.shape == (10, 6)
    
    f.close()
    shutil.rmtree(dir)

