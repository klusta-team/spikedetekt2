"""Object-oriented interface to an experiment's data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import re
from itertools import chain
from collections import OrderedDict

import numpy as np
import pandas as pd
import tables as tb

from selection import select, slice_to_indices
from spikedetekt2.dataio.kwik import (get_filenames, open_files, close_files,
    add_spikes)
from spikedetekt2.dataio.utils import convert_dtype
from spikedetekt2.utils.six import (iteritems, string_types, iterkeys, 
    itervalues, next)
from spikedetekt2.utils.wrap import wrap

class SpikeCache(object):
    def __init__(self, spike_clusters=None, cache_fraction=1.,
                 nspikes=None,
                 features_masks=None,
                 waveforms_raw=None,
                 waveforms_filtered=None):
        self.spike_clusters = spike_clusters
        self.cache_fraction = cache_fraction
        self.nspikes = nspikes
        self.features_masks = features_masks
        self.waveforms_raw = waveforms_raw
        self.waveforms_filtered = waveforms_filtered
        
        self.features_masks_cached = None
        self.cache_indices = None
        
        assert self.nspikes == len(self.spike_clusters)
        assert self.nspikes == self.features_masks.shape[0]
        assert self.nspikes == self.waveforms_raw.shape[0]
        assert self.nspikes == self.waveforms_filtered.shape[0]
        
        assert cache_fraction > 0
        
    def cache_features_masks(self, offset=0):
        k = np.clip(int(1. / self.cache_fraction), 1, self.nspikes)
        # Load and save subset in feature_masks.
        self.features_masks_cached = self.features_masks[offset::k,...]
        self.cache_indices = np.arange(self.nspikes)[offset::k,...]
        self.cache_size = len(self.cache_indices)
    
    def load_features_masks(self, fraction=None, clusters=None):
        """Load a subset of features & masks. 
        
        Arguments:
          * fraction: fraction of spikes to load from the cache.
          * clusters: if not None, load all features & masks of all spikes in 
            the selected clusters.
            
        """
        assert fraction is not None or clusters is not None
        
        # Cache susbet of features masks and save them in an array.
        if self.features_masks_cached is None:
            self.cache_features_masks()
        
        if clusters is None:
            offset = 0
            k = np.clip(int(1. / fraction), 1, self.cache_size)
            
            # Load and save subset from cache_feature_masks.
            loaded_features_masks = self.features_masks_cached[offset::k,...]
            loaded_indices = self.cache_indices[offset::k]
            return loaded_indices, loaded_features_masks
        else:
            # Find the indices of all spikes in the requested clusters
            indices = np.in1d(self.spike_clusters, clusters)
            if self.cache_fraction == 1.:
                return indices, self.features_masks_cached[indices,...]
            else:
                fm = np.empty((len(indices),) + self.features_masks.shape[1:], 
                              dtype=self.features_masks.dtype)
                for j, i in enumerate(indices):
                    fm[j:j+1,...] = self.features_masks[i:i+1,...]
                return indices, fm
           
    def load_waveforms(self, clusters=None, count=None, filtered=True):
        pass
        
        