"""Prototype of an object-oriented data API for kwiklib."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from six import range


# -----------------------------------------------------------------------------
# Group objects
# -----------------------------------------------------------------------------
class Group(object):
    _d = None
    
    def __init__(self, d=None):
        if d is not None:
            self._d = d
        else:
            self._d = {}
    
    def __next__(self):
        return self._d.__next__()
        
    def __iter__(self):
        return self._d.__iter__()
        
    def __getitem__(self, key):
        return self._d[key]
        
    def __setitem__(self, key, val):
        self._d[key] = val
        
    def append(self, val):
        return self._d.append(val)

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()
        
    def __getattr__(self, key):
        return super(Group, self).__getattr__(key)
        
    def __setattr__(self, key, val):
        super(Group, self).__setattr__(key, val)


class HDF5Group(object):
    def __init__(self, f, path):
        self._f = f
        self._path = path
    
    def __next__(self):
        pass
        
    def __iter__(self):
        pass
        
    def __getitem__(self, key):
        pass
        
    def __setitem__(self, key, val):
        pass
        
    def append(self, val):
        pass

    def keys(self):
        pass

    def values(self):
        pass

    def items(self):
        pass
        
    def __getattr__(self, key):
        pass
        
    def __setattr__(self, key, val):
        pass



        
# -----------------------------------------------------------------------------
# Experiment objects
# -----------------------------------------------------------------------------
class MockChannels(objects):
    def __init__(self, nspikes=None, nclusters=None, nchannels=None,
        maxcolors=None):
        self.name = [str(i) for i in range(nchannels)]
        self.klustaviewa = wrapdict({
            'color': [i % maxcolors for i in range(nchannels)],
            'visible': [True] * nchannels,
        })
        
    def __getitem__(self, index):
        return wrapdict({
            'name': self.klustaviewa['color'][index],
            'klustaviewa': wrapdict(self.klustaviewa, index),
        })
        
    def __getattr__(self, key):
        pass
        

class MockExperiment(object):
    def __init__(self, nspikes=None, nclusters=None, nchannels=None,
        maxcolors=30):
        self.spikes = MockSpikes(nspikes=nspikes, nclusters=nclusters,
            nchannels=nchannels,)
        
        self.channels = MockChannels(nspikes=nspikes, nclusters=nclusters,
            nchannels=nchannels, maxcolors=maxcolors)
        
        self.clusters = MockClusters(nclusters=nclusters,
            nchannels=nchannels, maxcolors=maxcolors)
        
        


# -----------------------------------------------------------------------------
# Some tests
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    g = Group({0: 'val0', 1: 'val1'})
    for i in g:
        print(i)
    g.aa = 0
    
if 0:
    exp = Experiment()
    
    
    
    # All spikes in the first shank.
    spikes = exp.shanks[0].spikes
    # For all spikes, the cluster.
    spike_clusters = spikes.clusters
    # The list of all clusters.
    clusters = exp.shanks[0].clusters
    # Cluster of spike #10: two ways of accessing the same bit of data.
    assert spikes[10].cluster == spike_clusters[10]
    # Get all spikes in cluster 3.
    indices = spike_clusters == 3
    spikes_in_3 = spikes[indices]
    # Get all spikes belong in one of the selected clusters.
    clusters_sel = [2, 3, 5, 7, 11]
    indices = np.in1d(spike_clusters, clusters_sel)
    myspikes = spikes[indices]
    
    