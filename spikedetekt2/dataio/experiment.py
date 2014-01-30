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
from spikedetekt2.dataio.kwik import (get_filenames, open_files, close_files
    )
from spikedetekt2.dataio.utils import convert_dtype
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
        if not obj:
            r = []
            return r
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

class ArrayProxy(object):
    """Proxy to a view of an array."""
    def __init__(self, arr, col=None):
        self._arr = arr
        self._col = col
    
    @property
    def shape(self):
        return self._arr.shape[:-1]
    
    def __getitem__(self, item):
        if self._col is None:
            return self._arr[item]
        else:
            if isinstance(item, tuple):
                item += (self._col,)
                return self._arr[item]
            else:
                return self._arr[item, ..., self._col]
        
def get_row_shape(arr):
    """Return the shape of a row of an array."""
    return (1,) + arr.shape[1:]
        
def empty_row(arr, dtype=None):
    """Create an empty row for a given array."""
    return np.zeros(get_row_shape(arr), dtype=arr.dtype)
        
        
# -----------------------------------------------------------------------------
# Node wrappers
# -----------------------------------------------------------------------------
class Node(object):
    def __init__(self, files, node=None, root=None):
        self._files = files
        self._kwik = self._files.get('kwik', None)
        assert self._kwik is not None
        if node is None:
            node = self._kwik.root
        self._node = node
        self._root = root
        
    def _gen_children(self, container_name=None, child_class=None):
        """Return a dictionary {child_id: child_instance}."""
        # The container with the children is either the current node, or
        # a child of this node.
        if container_name is None:
            container = self._node
        else:
            container = self._node._f_getChild(container_name)
        return OrderedDict([
            (_get_child_id(child), child_class(self._files, child, root=self._root))
                for child in container
            ])
    
    def _get_child(self, child_name):
        """Return the child specified by its name.
        If this child has a `hdf5_path` special, then the path is resolved,
        and the referred child in another file is returned.
        """
        child = self._node._f_getChild(child_name)
        try:
            # There's a link that needs to be resolved: return it.
            path = child._f_getAttr('hdf5_path')
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
        # Do not override if key is an attribute of this class.
        if key.startswith('_'):
            try:
                return self.__dict__[key]
            # Accept nodewrapper._method if _method is a method of the PyTables
            # Node object.
            except KeyError:
                return getattr(self._node, key)
        try:
            # Return the wrapped node if the child is a group.
            attr = getattr(self._node, key)
            if isinstance(attr, tb.Group):
                return NodeWrapper(attr)
            else:
                return attr
        # Return the attribute.
        except:
            try:
                return self._node._f_getAttr(key)
            except AttributeError:
                raise "{key} needs to be an attribute of {node}".format(
                    key=key, node=self._node._v_name)
            
    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
            return
        # Ensure the key is an existing attribute to the current node.
        try:
            self._node._f_getAttr(key)
        except AttributeError:
            raise "{key} needs to be an attribute of {node}".format(
                key=key, node=self._node._v_name)
        # Set the attribute.
        self._node._f_setAttr(key, value)
            
    def __dir__(self):
        return self._node.__dir__()
        
    def __repr__(self):
        return self._node.__repr__()

class DictVectorizer(object):
    """This object serves as a vectorized proxy for a dictionary of objects 
    that have individual fields of interest. For example: d={k: obj.attr1}.
    The object dv = DictVectorizer(d, 'attr1.subattr') can be used as:
    
        dv[3]
        dv[[1,2,5]]
        dv[2:4]
    
    """
    def __init__(self, dict, path):
        self._dict = dict
        self._path = path.split('.')
        
    def _get_path(self, key):
        """Resolve the path recursively for a given key of the dictionary."""
        val = self._dict[key]
        for p in self._path:
            val = getattr(val, p)
        return val
        
    def _set_path(self, key, value):
        """Resolve the path recursively for a given key of the dictionary,
        and set a value."""
        val = self._dict[key]
        for p in self._path[:-1]:
            val = getattr(val, p)
        setattr(val, key, value)
        
    def __getitem__(self, item):
        if isinstance(item, slice):
            item = slice_to_indices(item, lenindices=len(self._dict))
        if hasattr(item, '__len__'):
            return np.array([self._get_path(k) for k in item])
        else:
            return self._get_path(item)
            
    def __setitem__(self, item, value):
        if key.startswith('_'):
            self.__dict__[key] = value
            return
        if isinstance(item, slice):
            item = slice_to_indices(item, lenindices=len(self._dict))
        if hasattr(item, '__len__'):
            if not hasattr(value, '__len__'):
                value = [value] * len(item)
            for k, val in zip(item, value):
                self._set_path(k, value)
        else:
            return self._set_path(item, value)
        

# -----------------------------------------------------------------------------
# Experiment class and sub-classes.
# -----------------------------------------------------------------------------
class Experiment(Node):
    """An Experiment instance holds all information related to an
    experiment. One can access any information using a logical structure
    that is somewhat independent from the physical representation on disk.
    """
    def __init__(self, name=None, dir=None, files=None, mode='r', prm={}):
        """`name` must correspond to the basename of the files."""
        self.name = name
        self._dir = dir
        self._mode = mode
        self._files = files
        self._prm = prm
        if self._files is None:
            self._files = open_files(self.name, dir=self._dir, mode=self._mode)
        def _get_filename(file):
            if file is None:
                return None
            else:
                return os.path.realpath(file.filename)
        self._filenames = {type: _get_filename(file)
            for type, file in iteritems(self._files)}
        super(Experiment, self).__init__(self._files)
        self._root = self._node
        
        self.application_data = NodeWrapper(self._root.application_data)
        self.user_data = NodeWrapper(self._root.user_data)
        
        self.channel_groups = self._gen_children('channel_groups', ChannelGroup)
        self.recordings = self._gen_children('recordings', Recording)
        self.event_types = self._gen_children('event_types', EventType)
        
    def gen_filename(self, extension):
        if extension.startswith('.'):
            extension = extension[1:]
        return os.path.splitext(self._filenames['kwik'])[0] + '.' + extension
        
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
    def __init__(self, files, node=None, root=None):
        super(ChannelGroup, self).__init__(files, node, root=root)
        
        self.name = self._node._v_attrs.name
        self.adjacency_graph = self._node._v_attrs.adjacency_graph
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
        
        self.channels = self._gen_children('channels', Channel)
        self.clusters = ClustersNode(self._files, self._node.clusters)
        self.cluster_groups = ClusterGroupsNode(self._files, self._node.cluster_groups)
        
        self.spikes = Spikes(self._files, self._node.spikes)
        
class Spikes(Node):
    def __init__(self, files, node=None, root=None):
        super(Spikes, self).__init__(files, node, root=root)
        
        self.time_samples = self._node.time_samples
        self.time_fractional = self._node.time_fractional
        self.recording = self._node.recording
        self.clusters = Clusters(self._files, self._node.clusters)
        
        # Get large datasets, that may be in external files.
        self.features_masks = self._get_child('features_masks')
        self.waveforms_raw = self._get_child('waveforms_raw')
        self.waveforms_filtered = self._get_child('waveforms_filtered')
        
        self.nsamples, self.nchannels = self.waveforms_raw.shape[1:]
        self.features = ArrayProxy(self.features_masks, col=0)
        self.nfeatures = self.features.shape[1]
        self.masks = ArrayProxy(self.features_masks, col=1)
       
    def add(self, time_samples=None, time_fractional=0,
            recording=0, cluster=0, cluster_original=0,
            features_masks=None, features=None, masks=None,
            waveforms_raw=None, waveforms_filtered=None,
            ):
        """Add a spike. Only `time_samples` is mandatory."""
        if features_masks is None:
            # Default features and masks
            if features is None:
                features = np.zeros((1, self.nfeatures), dtype=np.float32)
            if masks is None:
                masks = np.zeros((1, self.nfeatures), dtype=np.float32)
            
            # Ensure features and masks have the right number of dimensions.
            # features.shape is (1, nfeatures)
            # masks.shape is however  (nchannels,)
            if features.ndim == 1:
                features = np.expand_dims(features, axis=0)
            if masks.ndim == 1:
                masks = np.expand_dims(masks, axis=0)
            
            # masks.shape is now    (1,nchannels,)
            # Tile the masks if needed: same mask value on each channel.
            if masks.shape[1] < features.shape[1]:
                nfeatures_per_channel = features.shape[1] // masks.shape[1]
                masks = np.repeat(masks, nfeatures_per_channel, axis = 1)
            # # masks.shape is (1, nfeatures) - what we want
            # Concatenate features and masks
            features_masks = np.dstack((features, masks))
            
            
        if waveforms_raw is None:
            waveforms_raw = empty_row(self.waveforms_raw)
        if waveforms_raw.ndim < 3:
            waveforms_raw = np.expand_dims(waveforms_raw, axis=0)
            
        if waveforms_filtered is None:
            waveforms_filtered = empty_row(self.waveforms_filtered)
        if waveforms_filtered.ndim < 3:
            waveforms_filtered = np.expand_dims(waveforms_filtered, axis=0)
            
        self.time_samples.append((time_samples,))
        self.time_fractional.append((time_fractional,))
        self.recording.append((recording,))
        self.clusters.main.append((cluster,))
        self.clusters.original.append((cluster_original,))
        self.features_masks.append(features_masks)
        self.waveforms_raw.append(convert_dtype(waveforms_raw, np.int16))
        self.waveforms_filtered.append(convert_dtype(waveforms_filtered, np.int16))
    
    def __getitem__(self, item):
        raise NotImplementedError("""It is not possible to select entire spikes 
            yet.""")
            
    def __len__(self):
        return self.time_samples.shape[0]
        
class Clusters(Node):
    def __init__(self, files, node=None, root=None):
        super(Clusters, self).__init__(files, node, root=root)        
        # Each child of the Clusters group is assigned here.
        for node in self._node._f_iterNodes():
            setattr(self, node._v_name, node)
        
class Clustering(Node):
    def __init__(self, files, node=None, root=None, child_class=None):
        super(Clustering, self).__init__(files, node, root=root)        
        self._dict = self._gen_children(child_class=child_class)
        self.color = DictVectorizer(self._dict, 'application_data.klustaviewa.color')

    def __getitem__(self, item):
        return self._dict[item]
        
    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def __contains__(self, v):
        return v in self._dict

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def iteritems(self):
        return self._dict.iteritems()
        
class ClustersClustering(Clustering):
    def __init__(self, *args, **kwargs):
        super(ClustersClustering, self).__init__(*args, **kwargs)
        self.group = DictVectorizer(self._dict, 'cluster_group')
        
class ClusterGroupsClustering(Clustering):
    pass
        
class ClustersNode(Node):
    def __init__(self, files, node=None, root=None):
        super(ClustersNode, self).__init__(files, node, root=root)        
        # Each child of the group is assigned here.
        for node in self._node._f_iterNodes():
            setattr(self, node._v_name, ClustersClustering(self._files, node, child_class=Cluster))
        
class ClusterGroupsNode(Node):
    def __init__(self, files, node=None, root=None):
        super(ClusterGroupsNode, self).__init__(files, node, root=root)        
        # Each child of the group is assigned here.
        for node in self._node._f_iterNodes():
            setattr(self, node._v_name, ClusterGroupsClustering(self._files, node, child_class=ClusterGroup))
        
class Channel(Node):
    def __init__(self, files, node=None, root=None):
        super(Channel, self).__init__(files, node, root=root)
        
        self.name = self._node._v_attrs.name
        self.kwd_index = self._node._v_attrs.kwd_index
        self.ignored = self._node._v_attrs.ignored
        self.position = self._node._v_attrs.position
        self.voltage_gain = self._node._v_attrs.voltage_gain
        self.display_threshold = self._node._v_attrs.display_threshold
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
    
class Cluster(Node):
    def __init__(self, files, node=None, root=None):
        super(Cluster, self).__init__(files, node, root=root)
        
        self.cluster_group = self._node._v_attrs.cluster_group
        self.mean_waveform_raw = self._node._v_attrs.mean_waveform_raw
        self.mean_waveform_filtered = self._node._v_attrs.mean_waveform_filtered
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
        self.quality_measures = NodeWrapper(self._node.quality_measures)

class ClusterGroup(Node):
    def __init__(self, files, node=None, root=None):
        super(ClusterGroup, self).__init__(files, node, root=root)
        self.name = self._node._v_attrs.name
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
    
class Recording(Node):
    def __init__(self, files, node=None, root=None):
        super(Recording, self).__init__(files, node, root=root)
        
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
    def __init__(self, files, node=None, root=None):
        super(EventType, self).__init__(files, node, root=root)
    
        self.events = Events(self._files, self._node.events)
        
        self.application_data = NodeWrapper(self._node.application_data)
        self.user_data = NodeWrapper(self._node.user_data)
    
class Events(Node):
    def __init__(self, files, node=None, root=None):
        super(Events, self).__init__(files, node, root=root)
        
        self.time_samples = self._node.time_samples
        self.recording = self._node.recording
        
        self.user_data = NodeWrapper(self._node.user_data)
