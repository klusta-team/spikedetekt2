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

from spikedetekt2.dataio.utils import save_json
from spikedetekt2.utils.six import iteritems

# Disable PyTables' NaturalNameWarning due to nodes which have names starting 
# with an integer.
warnings.simplefilter('ignore', tb.NaturalNameWarning)


# -----------------------------------------------------------------------------
# KWIK file creation
# -----------------------------------------------------------------------------
def create_kwik(path, experiment_name=None, prm=None, prb=None):
    """Create a KWIK file.
    
    Arguments:
      * path: path to the .kwik file.
      * experiment_name
      * prm: a dictionary representing the contents of the PRM file (used for
        SpikeDetekt)
    
    """
    if experiment_name is None:
        experiment_name = ''
    if prm is None:
        prm = {}
    if prb is None:
        prb = {}
    
    file = tb.openFile(path, mode='w')
    
    file.root._v_attrs.kwik_version = 2
    file.root._v_attrs.name = experiment_name

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
        group._v_attrs.name = 'channel_group_{0:d}'.format(igroup)
        group._v_attrs.adjacency_graph = group_info.get('graph', np.zeros((0, 2)))
        file.createGroup(group, 'application_data')
        file.createGroup(group, 'user_data')
        
        # Create channels.
        file.createGroup(group, 'channels')
        channels = group_info.get('channels', [])
        for channel_idx in channels:
            # channel is the absolute channel index.
            channel = file.createGroup(group.channels, str(channel_idx))
            channel._v_attrs.name = 'channel_{0:d}'.format(channel_idx)
            
            ############### TODO
            channel._v_attrs.kwd_index = 0
            channel._v_attrs.ignored = False
            channel._v_attrs.position = group_info.get('geometry', {}). \
                get(channel_idx, None)
            channel._v_attrs.voltage_gain = 0.
            channel._v_attrs.display_threshold = 0.
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
        fm._v_attrs.hdf5_path = '{{KWX}}/channel_groups/{0:d}/features_masks'. \
            format(igroup)
        wr = file.createGroup(spikes, 'waveforms_raw')
        wr._v_attrs.hdf5_path = '{{KWX}}/channel_groups/{0:d}/waveforms_raw'. \
            format(igroup)
        wf = file.createGroup(spikes, 'waveforms_filtered')
        wf._v_attrs.hdf5_path = '{{KWX}}/channel_groups/{0:d}/waveforms_filtered'. \
            format(igroup)
        
        # Create clusters.
        file.createGroup(group, 'clusters')
        
        # Create cluster groups.
        file.createGroup(group, 'cluster_groups')
        
    # Create recordings.
    file.createGroup('/', 'recordings')
    
    # Create event types.
    file.createGroup('/', 'event_types')
            
    file.close()
    
    
# -----------------------------------------------------------------------------
# HDF5 files creation
# -----------------------------------------------------------------------------
def create_kwx(path, channel_groups=None, nwavesamples=None, nfeatures=None,
               nchannels=None, has_masks=True):
    """Create an empty KWX file.
    
    Arguments:
      * channel_groups: a dictionary 'ichannel_group': 'channel_group_info'
        where channel_group_info is a dictionary with the optional fields:
          * nchannels
          * nwavesamples
          * nfeatures
      * nwavesamples (common to all channel groups if set)
      * nfeatures (total number of features per spike, common to all channel groups if set)
      * nchannels (number of channels per channel group, common to all channel groups if set)
    
    """
    file = tb.openFile(path, mode='w')
    file.createGroup('/', 'channel_groups')
    
    for ichannel_group, channel_group_info in enumerate(channel_groups):
        nchannels_ = channel_group_info.get('nchannels', nchannels)
        nfeatures_ = channel_group_info.get('nfeatures', nfeatures)
        nwavesamples_ = channel_group_info.get('nwavesamples', nwavesamples)
        
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
   
def create_kwe(path, ):
    """Create an empty KWE file."""
    file = tb.openFile(path, mode='w')
    
    # Create the tables.
    file.createTable('/', 'events',
                     get_events_description())
    # file.createTable('/', 'event_types',
                     # get_event_types_description())
                                                   
    file.close()
           
    