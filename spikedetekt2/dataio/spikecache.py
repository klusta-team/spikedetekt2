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
        self.cache_features_masks = None
        
        assert self.nspikes == len(self.spike_clusters)
        assert self.nspikes == self.features_masks.shape[0]
        assert self.nspikes == self.waveforms_raw.shape[0]
        assert self.nspikes == self.waveforms_filtered.shape[0]
        
    def cache_features_masks(self, offset=0):
        
        self.cache_features_masks = self.features_masks[offset::k]
    
    def load_features_masks(self, fraction=None, clusters=None):
        """Load a subset of features & masks. 
        
        Arguments:
          * fraction: fraction of spikes to load from the cache.
          * clusters: if not None, load all features & masks of all spikes in 
            the selected clusters.
            
        """
        assert fraction is not None
        
           
    def load_waveforms(self, clusters=None, count=None):
        pass
        
        