"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import warnings
from collections import OrderedDict, Iterable

import numpy as np
import tables as tb

from spikedetekt2.utils.six import itervalues, iteritems, string_types

# Disable PyTables' NaturalNameWarning due to nodes which have names starting 
# with an integer.
warnings.simplefilter('ignore', tb.NaturalNameWarning)


# -----------------------------------------------------------------------------
# File names
# -----------------------------------------------------------------------------
RAW_TYPES = ('raw.kwd', 'high.kwd', 'low.kwd')
FILE_TYPES = ('kwik', 'kwx') + RAW_TYPES

def get_filenames(name, dir=None):
    """Generate a list of filenames for the different files in a given 
    experiment, which name is given."""
    if dir is None:
        dir = os.path.dirname(os.path.realpath(__file__))
    name = os.path.splitext(name)[0]
    return {type: os.path.join(dir, name + '.' + type) for type in FILE_TYPES}
    
def get_basename(path):
    bn = os.path.basename(path)
    bn = os.path.splitext(bn)[0]
    if bn.split('.')[-1] in ('raw', 'high', 'low'):
        return os.path.splitext(bn)[0]
    else:
        return bn


# -----------------------------------------------------------------------------
# Opening/closing functions
# -----------------------------------------------------------------------------
def open_file(path, mode=None):
    if mode is None:
        mode = 'r'
    try:
        return tb.openFile(path, mode)
    except:
        return None

def open_files(name, dir=None, mode=None):
    filenames = get_filenames(name, dir=dir)
    return {type: open_file(filename, mode=mode) 
            for type, filename in iteritems(filenames)}

def close_files(name, dir=None):
    if isinstance(name, string_types):
        filenames = get_filenames(name, dir=dir)
        files = [open_file(filename) for filename in itervalues(filenames)]
    else:
        files = itervalues(name)
    [file.close() for file in files]
    

# -----------------------------------------------------------------------------
# HDF5 file creation
# -----------------------------------------------------------------------------
def create_kwik(path, experiment_name=None, prm=None, prb=None):
    """Create a KWIK file.
    
    Arguments:
      * path: path to the .kwik file.
      * experiment_name
      * prm: a dictionary representing the contents of the PRM file (used for
        SpikeDetekt)
      * prb: a dictionary with the contents of the PRB file
    
    """
    if experiment_name is None:
        experiment_name = ''
    if prm is None:
        prm = {}
    if prb is None:
        prb = {}
    
    file = tb.openFile(path, mode='w')
    
    file.root._f_setattr('kwik_version', 2)
    file.root._f_setattr('name', experiment_name)

    file.createGroup('/', 'application_data')
    
    # Set the SpikeDetekt parameters
    file.createGroup('/application_data', 'spikedetekt')
    for prm_name, prm_value in iteritems(prm):
        setattr(file.root.application_data.spikedetekt,
                prm_name,
                prm_value)
    
    file.createGroup('/', 'user_data')
    
    # Create channel groups.
    file.createGroup('/', 'channel_groups')
    for igroup, group_info in enumerate(prb.get('channel_groups', [])):
        group = file.createGroup('/channel_groups', str(igroup))
        # group_info: channel, graph, geometry
        group._f_setattr('name', 'channel_group_{0:d}'.format(igroup))
        group._f_setattr('adjacency_graph', group_info.get('graph', np.zeros((0, 2))))
        file.createGroup(group, 'application_data')
        file.createGroup(group, 'user_data')
        
        # Create channels.
        file.createGroup(group, 'channels')
        channels = group_info.get('channels', [])
        for channel_idx in channels:
            # channel is the absolute channel index.
            channel = file.createGroup(group.channels, str(channel_idx))
            channel._f_setattr('name', 'channel_{0:d}'.format(channel_idx))
            
            ############### TODO
            channel._f_setattr('kwd_index', 0)
            channel._f_setattr('ignored', False)
            channel._f_setattr('position', group_info.get('geometry', {}). \
                get(channel_idx, None))
            channel._f_setattr('voltage_gain', 0.)
            channel._f_setattr('display_threshold', 0.)
            file.createGroup(channel, 'application_data')
            file.createGroup(channel.application_data, 'spikedetekt')
            file.createGroup(channel.application_data, 'klustaviewa')
            file.createGroup(channel, 'user_data')
            
        # Create spikes.
        spikes = file.createGroup(group, 'spikes')
        file.createEArray(spikes, 'time_samples', tb.UInt64Atom(), (0,))
        file.createEArray(spikes, 'time_fractional', tb.UInt8Atom(), (0,))
        file.createEArray(spikes, 'recording', tb.UInt16Atom(), (0,))
        file.createEArray(spikes, 'cluster', tb.UInt32Atom(), (0,))
        file.createEArray(spikes, 'cluster_original', tb.UInt32Atom(), (0,))
        
        fm = file.createGroup(spikes, 'features_masks')
        fm._f_setattr('hdf5_path', '{{kwx}}/channel_groups/{0:d}/features_masks'. \
            format(igroup))
        wr = file.createGroup(spikes, 'waveforms_raw')
        wr._f_setattr('hdf5_path', '{{kwx}}/channel_groups/{0:d}/waveforms_raw'. \
            format(igroup))
        wf = file.createGroup(spikes, 'waveforms_filtered')
        wf._f_setattr('hdf5_path', '{{kwx}}/channel_groups/{0:d}/waveforms_filtered'. \
            format(igroup))
        
        # Create clusters.
        file.createGroup(group, 'clusters')
        
        # Create cluster groups.
        file.createGroup(group, 'cluster_groups')
        
    # Create recordings.
    file.createGroup('/', 'recordings')
    
    # Create event types.
    file.createGroup('/', 'event_types')
            
    file.close()

def create_kwx(path, prb=None, prm=None, has_masks=True):
    """Create an empty KWX file.
    
    Arguments:
      * prb: the PRB dictionary
      * nwavesamples (common to all channel groups if set)
      * nfeatures (total number of features per spike, common to all channel groups if set)
      * nchannels (number of channels per channel group, common to all channel groups if set)
    
    """
    
    if prb is None:
        prb = {}
    if prm is None:
        prm = {}
    
    nchannels = prm.get('nchannels', None)
    nfeatures = prm.get('nfeatures', None)
    nwavesamples = prm.get('nwavesamples', None)
        
    file = tb.openFile(path, mode='w')
    file.createGroup('/', 'channel_groups')
    
    for ichannel_group, chgrp_info in enumerate(prb.get('channel_groups', [])):
        nchannels_ = len(chgrp_info.get('channels', [])) or nchannels or 0
        nfeatures_ = chgrp_info.get('nfeatures', nfeatures) or 0
        nwavesamples_ = chgrp_info.get('nwavesamples', nwavesamples) or 0
        
        assert nchannels_ > 0
        assert nfeatures_ > 0
        assert nwavesamples_ > 0
        
        channel_group_path = '/channel_groups/{0:d}'.format(ichannel_group)
        
        # Create the HDF5 group for each channel group.
        file.createGroup('/channel_groups', 
                         '{0:d}'.format(ichannel_group))
                         
        # Create the tables.
        if has_masks:
            # Features + masks.
            file.createEArray(channel_group_path, 'features_masks',
                              tb.Float32Atom(), (0, nfeatures_, 2))
        else:
            file.createEArray(channel_group_path, 'features_masks',
                              tb.Float32Atom(), (0, nfeatures_))
        
        file.createEArray(channel_group_path, 'waveforms_raw',
                          tb.Int16Atom(), (0, nwavesamples_, nchannels_))
        file.createEArray(channel_group_path, 'waveforms_filtered',
                          tb.Int16Atom(), (0, nwavesamples_, nchannels_))
                                                   
    file.close()
            
def create_kwd(path, type='raw', nchannels_tot=None, recordings=None,):
    """Create an empty KWD file.
    
    Arguments:
      * type: 'raw', 'high', or 'low'
      * nchannels_tot: total number of channels
      * recordings: a dictionary irecording: recording_info where 
        recording_info is a dictionary with the optional fields:
          * nsamples: expected number of samples in that recording
    
    """
    if recordings is None:
        recordings = []
        
    file = tb.openFile(path, mode='w')
    
    for irecording, recording_info in enumerate(recordings):
        nsamples_ = recording_info.get('nsamples', None)
        
        file.createGroup('/', 'recording{0:d}'.format(irecording))    
        recording_path = '/recording{0:d}'.format(irecording)
        
        file.createEArray(recording_path, 'data_{0:s}'.format(type), 
                          tb.Int16Atom(), 
                          (0, nchannels_tot), expectedrows=nsamples_)
    
    file.close()

def create_files(name, dir=None, prm=None, prb=None):
    
    # TODO: retrieve nwavesamples, nfeatures, nchannels
    # nchannels_tot
    # from PRM/PRB to pass them to the create_* functions.
    
    filenames = get_filenames(name, dir=dir)
    
    create_kwik(filenames['kwik'], prm=prm, prb=prb)
    create_kwx(filenames['kwx'], prb=prb, prm=prm)
    
    create_kwd(filenames['raw.kwd'], 'raw')
    create_kwd(filenames['high.kwd'], 'high')
    create_kwd(filenames['low.kwd'], 'low')
    
    return filenames

    
# -----------------------------------------------------------------------------
# Adding items in the files
# -----------------------------------------------------------------------------
def add_recording(fd, id=None, name=None, sample_rate=None, start_time=None, 
                  start_sample=None, bit_depth=None, band_high=None,
                  band_low=None):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    kwik = fd.get('kwik', None)
    # The KWIK needs to be there.
    assert kwik is not None
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        recordings = sorted([n._v_name 
                             for n in kwik.listNodes('/recordings')])
        if recordings:
            id = str(max([int(r) for r in recordings if r.isdigit()]) + 1)
        else:
            id = '0'
    # Default name: recording_X if X is an integer, or the id.
    if name is None:
        if id.isdigit():
            name = 'recording_{0:s}'.format(id)
        else:
            name = id
    recording = kwik.createGroup('/recordings', id)
    recording._f_setattr('name', name)
    recording._f_setattr('start_time', start_time)
    recording._f_setattr('start_sample', start_sample)
    recording._f_setattr('sample_rate', sample_rate)
    recording._f_setattr('bit_depth', bit_depth)
    recording._f_setattr('band_high', band_high)
    recording._f_setattr('band_low', band_low)
    
    kwik_raw = kwik.createGroup('/recordings/' + id, 'raw')
    kwik_high = kwik.createGroup('/recordings/' + id, 'high')
    kwik_low = kwik.createGroup('/recordings/' + id, 'low')
    
    kwik_raw._f_setattr('hdf5_path', '{raw.kwd}/recordings/' + id)
    kwik_high._f_setattr('hdf5_path', '{high.kwd}/recordings/' + id)
    kwik_low._f_setattr('hdf5_path', '{low.kwd}/recordings/' + id)
    
    kwik.createGroup('/recordings/' + id, 'user_data')
        
    for type in RAW_TYPES:
        kwd = fd.get(type, None)
        if kwd:
            # TODO
            pass
    
def add_event_type(fd, id=None, evt=None):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    kwik = fd.get('kwik', None)
    # The KWIK needs to be there.
    assert kwik is not None
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        event_types = sorted([n._v_name 
                             for n in kwik.listNodes('/event_types')])
        if event_types:
            id = str(max([int(r) for r in event_types if r.isdigit()]) + 1)
        else:
            id = '0'
    event_type = kwik.createGroup('/event_types', id)
    
    kwik.createGroup(event_type, 'user_data')
    
    app = kwik.createGroup(event_type, 'application_data')
    kv = kwik.createGroup(app, 'klustaviewa')
    kv._f_setattr('color', None)
    
    events = kwik.createGroup(event_type, 'events')
    kwik.createEArray(events, 'time_samples', tb.UInt64Atom(), (0,))
    kwik.createEArray(events, 'recording', tb.UInt16Atom(), (0,))
    kwik.createGroup(events, 'user_data')
    
def add_cluster(fd, channel_group_id=None, id=None, 
    cluster_group=None,
    mean_waveform_raw=None,
    mean_waveform_filtered=None,
    ):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    kwik = fd.get('kwik', None)
    # The KWIK needs to be there.
    assert kwik is not None
    # The channel group id containing the new cluster group must be specified.
    assert channel_group_id is not None
    clusters_path = '/channel_groups/{0:s}/clusters'.format(channel_group_id)
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        clusters = sorted([n._v_name 
                             for n in kwik.listNodes(clusters_path)])
        if clusters:
            id = str(max([int(r) for r in clusters if r.isdigit()]) + 1)
        else:
            id = '0'
    cluster = kwik.createGroup(clusters_path, id)
    
    cluster._f_setattr('cluster_group', cluster_group)
    cluster._f_setattr('mean_waveform_raw', mean_waveform_raw)
    cluster._f_setattr('mean_waveform_filtered', mean_waveform_filtered)
    
    # TODO
    quality = kwik.createGroup(cluster, 'quality_measures')
    quality._f_setattr('isolation_distance', None)
    quality._f_setattr('matrix_isolation', None)
    quality._f_setattr('refractory_violation', None)
    quality._f_setattr('amplitude', None)
    
    kwik.createGroup(cluster, 'user_data')
    
    app = kwik.createGroup(cluster, 'application_data')
    kv = kwik.createGroup(app, 'klustaviewa')
    kv._f_setattr('color', None)
    
def add_cluster_group(fd, channel_group_id=None, id=None, name=None):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    kwik = fd.get('kwik', None)
    # The KWIK needs to be there.
    assert kwik is not None
    # The channel group id containing the new cluster group must be specified.
    assert channel_group_id is not None
    cluster_groups_path = '/channel_groups/{0:s}/cluster_groups'.format(channel_group_id)
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        cluster_groups = sorted([n._v_name 
                             for n in kwik.listNodes(cluster_groups_path)])
        if cluster_groups:
            id = str(max([int(r) for r in cluster_groups if r.isdigit()]) + 1)
        else:
            id = '0'
    # Default name: cluster_group_X if X is an integer, or the id.
    if name is None:
        if id.isdigit():
            name = 'cluster_group_{0:s}'.format(id)
        else:
            name = id
    cluster_group = kwik.createGroup(cluster_groups_path, id)
    cluster_group._f_setattr('name', name)
    
    kwik.createGroup(cluster_group, 'user_data')
    
    app = kwik.createGroup(cluster_group, 'application_data')
    kv = kwik.createGroup(app, 'klustaviewa')
    kv._f_setattr('color', None)
    
    
