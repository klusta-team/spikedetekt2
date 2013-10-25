"""Manage experiments."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import re
from itertools import chain

import numpy as np
import pandas as pd
import tables as tb

from selection import select
from spikedetekt2.dataio.kwik import (get_filenames, open_files, close_files
    )
from spikedetekt2.utils.six import (iteritems, string_types, iterkeys, 
    itervalues, next)
from spikedetekt2.utils.wrap import wrap


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _resolve_hdf5_path(files, path):
    """Resolve a HDF5 external link. Return the referred node (group or 
    dataset), or None if it does not exist.
    
    Arguments:
      * files: a dict {type: file_handle}.
      * path: a string like "{type}/path/to/node" where `type` is one of
        `kwx`, `raw.kwd`, etc.
      
    """  
    nodes = path.split('/')
    path_ext = '/' + '/'.join(nodes[1:])
    type = nodes[0]
    pattern = r'\{([a-zA-Z\._]+)\}'
    assert re.match(pattern, type)
    r = re.search(pattern, type)
    assert r
    type = r.group(1)
    # Resolve the link.
    file = files.get(type, None)
    if file:
        return file.getNode(path_ext)
    else:
        return None
    
def _get_child_id(child):
    id = child._v_name
    if id.isdigit():
        return int(id)
    else:
        return id

def _print_instance(obj, depth=0, name=''):
    # Handle the first element of the list/dict.
    if isinstance(obj, (list, dict)):
        if isinstance(obj, list):
            sobj = obj[0]
            key = '0'
        elif isinstance(obj, dict):
            key, sobj = next(iteritems(obj))
        if isinstance(sobj, (list, dict, int, long, string_types, np.ndarray, 
                      float)):
            r = []
        else:
            r = [(depth+1, str(key))] + _print_instance(sobj, depth+1)
    # Arrays do not have children.
    elif isinstance(obj, (np.ndarray, tb.EArray)):
        r = []
    # Handle class instances.
    elif hasattr(obj, '__dict__'):
        fields = {k: v 
            for k, v in iteritems(vars(obj)) 
                if not k.startswith('_')}
        r = list(chain(*[_print_instance(fields[n], depth=depth+1, name=str(n)) 
                for n in sorted(iterkeys(fields))]))
    else:
        r = []
    # Add the current object's display string.
    if name:
        if isinstance(obj, tb.EArray):
            s = name + ' [{dtype} {shape}]'.format(dtype=obj.dtype, 
                shape=obj.shape)
        else:
            s = name
        r = [(depth, s)] + r
    return r

        
# -----------------------------------------------------------------------------
# Node wrappers
# -----------------------------------------------------------------------------
class Node(object):
    def __init__(self, files, node=None):
        self._files = files
        self._kwik = self._files.get('kwik', None)
        assert self._kwik is not None
        if node is None:
            node = self._kwik.root
        self._node = node
        
    def _gen_children(self, container_name, child_class):
        """Return a dictionary {child_id: child_instance}."""
        return {
            _get_child_id(child): child_class(self._files, child)
                for child in self._node._f_getChild(container_name)
            }
    
    def _get_child(self, child_name):
        """Return the child specified by its name.
        If this child has a `hdf5_path` special, then the path is resolved,
        and the referred child in another file is returned.
        """
        child = self._node._f_getChild(child_name)
        try:
            # There's a link that needs to be resolved: return it.
            path = child._f_getattr('hdf5_path')
            return _resolve_hdf5_path(self._files, path)
        except AttributeError:
            # No HDF5 external link: just return the normal child.
            return child

class NodeWrapper(object):
    """Like a PyTables node, but supports in addition: `node.attr`."""
    def __init__(self, node):
        self._node = node
        
    def __getitem__(self, key):
        return self._node[key]
        
    def __getattr__(self, key):
        try:
            attr = getattr(self._node, key)
            if isinstance(attr, tb.Group):
                return NodeWrapper(attr)
            else:
                return attr
        except:
            return self._node._f_getAttr(key)
            
    def __dir__(self):
        return self._node.__dir__()
        
    def __repr__(self):
        return self._node.__repr__()

        
# -----------------------------------------------------------------------------
# Experiment class and sub-classes.
# -----------------------------------------------------------------------------
class Experiment(Node):
    """An Experiment instance holds all information related to an
    experiment. One can access any information using a logical structure
    that is somewhat independent from the physical representation on disk.
    """
    def __init__(self, name=None, dir=None, files=None, mode='r'):
        """`name` must correspond to the basename of the files."""
        self.name = name
        self._dir = dir
        self._mode = mode
        self._files = files
        if self._files is None:
            self._files = open_files(self.name, dir=self._dir, mode=self._mode)
        self._filenames = {type: os.path.realpath(file.filename)
            for type, file in iteritems(self._files)}
        super(Experiment, self).__init__(self._files)
        self._root = self._node
        
        self.application_data = NodeWrapper(self._root.application_data)
        self.user_data = NodeWrapper(self._root.user_data)
        
        self.channel_groups = self._gen_children('channel_groups', ChannelGroup)
        self.recordings = self._gen_children('recordings', Recording)
        self.event_types = self._gen_children('event_types', EventType)
        
    def __enter__(self):
        return self
    
    def close(self):
        if self._files is not None:
            close_files(self._files)
    
    def __repr__(self):
        n = "<Experiment '{name}'>".format(name=self.name)
        l = _print_instance(self, name=n)
        # print l
        return '\n'.join('    '*d + s for d, s in l)
    
    def __exit__ (self, type, value, tb):
        self.close()
        
class ChannelGroup(Node):
    def __init__(self, files, node=None):
        super(ChannelGroup, self).__init__(files, node)
        
        self.name = self._node._v_attrs.name
        self.adjacency_graph = self._node._v_attrs.adjacency_graph
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
        
        self.channels = self._gen_children('channels', Channel)
        self.clusters = self._gen_children('clusters', Cluster)
        self.cluster_groups = self._gen_children('cluster_groups', ClusterGroup)
        
        self.spikes = Spikes(self._files, self._node.spikes)
        
class Spikes(Node):
    def __init__(self, files, node=None):
        super(Spikes, self).__init__(files, node)
        
        self.time_samples = self._node.time_samples
        self.time_fractional = self._node.time_fractional
        self.recording = self._node.recording
        self.cluster = self._node.cluster
        self.cluster_original = self._node.cluster_original
        
        # Get large datasets, that may be in external files.
        self.features_masks = self._get_child('features_masks')
        self.waveforms_raw = self._get_child('waveforms_raw')
        self.waveforms_filtered = self._get_child('waveforms_filtered')
       
class Channel(Node):
    def __init__(self, files, node=None):
        super(Channel, self).__init__(files, node)
        
        self.name = self._node._v_attrs.name
        self.kwd_index = self._node._v_attrs.kwd_index
        self.ignored = self._node._v_attrs.ignored
        self.position = self._node._v_attrs.position
        self.voltage_gain = self._node._v_attrs.voltage_gain
        self.display_threshold = self._node._v_attrs.display_threshold
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
    
class Cluster(Node):
    def __init__(self, files, node=None):
        super(Cluster, self).__init__(files, node)
        
        self.cluster_group = self._node._v_attrs.cluster_group
        self.mean_waveform_raw = self._node._v_attrs.mean_waveform_raw
        self.mean_waveform_filtered = self._node._v_attrs.mean_waveform_filtered
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
        self.quality_measures = NodeWrapper(self._node.quality_measures)

class ClusterGroup(Node):
    def __init__(self, files, node=None):
        super(ClusterGroup, self).__init__(files, node)
        self.name = self._node._v_attrs.name
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
    
class Recording(Node):
    def __init__(self, files, node=None):
        super(Recording, self).__init__(files, node)
        
        self.name = self._node._v_attrs.name
        self.start_time = self._node._v_attrs.start_time
        self.start_sample = self._node._v_attrs.start_sample
        self.sample_rate = self._node._v_attrs.sample_rate
        self.bit_depth = self._node._v_attrs.bit_depth
        self.band_high = self._node._v_attrs.band_high
        self.band_low = self._node._v_attrs.band_low
        
        self.raw = self._get_child('raw')
        self.high = self._get_child('high')
        self.low = self._get_child('low')
        
        self.user_data = NodeWrapper(self._node.user_data)
    
class EventType(Node):
    def __init__(self, files, node=None):
        super(EventType, self).__init__(files, node)
    
        self.events = Events(self._files, self._node.events)
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
    
class Events(Node):
    def __init__(self, files, node=None):
        super(Events, self).__init__(files, node)
        
        self.time_samples = self._node.time_samples
        self.recording = self._node.recording
        
        self.user_data = NodeWrapper(self._node.user_data)