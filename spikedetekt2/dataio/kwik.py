"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables as tb
from collections import OrderedDict, Iterable

import numpy as np

from spikedetekt2.utils.six import iteritems


# -----------------------------------------------------------------------------
# Table descriptions
# -----------------------------------------------------------------------------
def get_spiketrain_description():
    return OrderedDict([
        ('sample', tb.Float64Col()),
        ('fractional', tb.UInt8Col()),
        ('recording', tb.UInt16Col()),
        ('cluster', tb.UInt32Col()),
        ])
        
def get_spikesorting_description(nfeatures=None):
    return OrderedDict([
        ('features', tb.Float32Col(shape=(nfeatures,))),
        ('masks', tb.UInt8Col(shape=(nfeatures,))),
        ('cluster_original', tb.UInt32Col()),
        ])
    
def get_waveforms_description(nwavesamples=None, nchannels=None):
    return OrderedDict([
        ('waveform_filtered', tb.Int16Col(shape=(nwavesamples*nchannels))),
        ('waveform_raw', tb.Int16Col(shape=(nwavesamples*nchannels))),
        ])

def get_events_description():
    return OrderedDict([
        ('sample', tb.UInt64Col()),
        ('recording', tb.UInt16Col()),
        ('event_type', tb.UInt16Col()),
        ])

def get_event_types_description():
    return OrderedDict([
        ('name', tb.StringCol(256)),
        ])


# -----------------------------------------------------------------------------
# KWIK file creation
# -----------------------------------------------------------------------------
def create_kwik(name=None, channel_groups=None, recordings=None,
                event_types=None):
    if channel_groups is None:
        channel_groups = []
    if recordings is None:
        recordings = []
    if event_types is None:
        event_types = []
        
    assert isinstance(channel_groups, Iterable)
    assert isinstance(recordings, Iterable)
    assert isinstance(event_types, Iterable)
        
    kwik = OrderedDict()
    kwik['VERSION'] = 2
    kwik['name'] = name
    kwik['application_data'] = {'spikedetekt': {}}
    kwik['user_data'] = {}
    kwik['channel_groups'] = channel_groups
    kwik['recordings'] = recordings
    kwik['events'] = {'hdf5_path': '{{KWE}}/events'}
    kwik['event_types'] = event_types
    return kwik
    
def create_kwik_channel_group(ichannel_group=None, name=None, graph=None,
    channels=None, cluster_groups=None):
    if channels is None:
        channels = []
    if cluster_groups is None:
        cluster_groups = []
        
    assert isinstance(channels, Iterable)
    assert isinstance(cluster_groups, Iterable)
    
    o = OrderedDict()
    o['name'] = name
    o['graph'] = graph
    o['application_data'] = {}
    o['user_data'] = {}
    o['channels'] = channels
    o['spikes'] = {
    'hdf5_path': {
    'spiketrain': '{{KWX}}/channel_groups/channel_group{0:d}/spiketrain'. \
        format(ichannel_group),
    'spikesorting': '{{KWX}}/channel_groups/channel_group{0:d}/spikesorting'. \
        format(ichannel_group),
    'waveforms': '{{KWX}}/channel_groups/channel_group{0:d}/waveforms'. \
        format(ichannel_group),
    }}
    o['cluster_groups'] = cluster_groups
    return o
    
def create_kwik_channel(name=None, ignored=False, position=None,
                        voltage_gain=None, display_threshold=None):
    o = OrderedDict()
    o['name'] = name
    o['ignored'] = ignored
    o['position'] = position
    o['voltage_gain'] = voltage_gain
    o['display_threshold'] = display_threshold
    o['application_data'] = {
        'klustaview': {},
        'spikedetekt': {},
    }
    o['user_data'] = {}
    return o
    
def create_kwik_cluster(color=None):
    o = OrderedDict()
    o['application_data'] = {
        'klustaviewa': {'color': color},
    }
    return o
    
def create_kwik_cluster_group(color=None, name=None, clusters=None):
    if clusters is None:
        clusters = []
        
    assert isinstance(clusters, Iterable)

    o = OrderedDict()
    o['name'] = name
    o['application_data'] = {
        'klustaviewa': {'color': color},
    }
    o['user_data'] = {}
    o['clusters'] = clusters
    return o
    
def create_kwik_recording(irecording=None, start_time=None,
                          name=None,
                          start_sample=None, sample_rate=None,
                          band_low=None, band_high=None, bit_depth=None):
    o = OrderedDict()
    o['name'] = name
    o['user_data'] = {}
    o['data'] = {
        'hdf5_path': 
        {
            'raw': '{{KWD_RAW}}/data_raw/recording{0:d}'. \
                format(irecording),
            'high_pass': '{{KWD_HIGH}}/data_high/recording{0:d}'. \
                format(irecording),
            'low_pass': '{{KWD_LOW}}/data_low/recording{0:d}'. \
                format(irecording),
        }
    }
    o['start_time'] = start_time
    o['start_sample'] = start_sample
    o['sample_rate'] = sample_rate
    o['band_low'] = band_low
    o['band_high'] = band_high
    o['bit_depth'] = bit_depth
    return o
       
def create_kwik_event_type(color=None):
    o = OrderedDict()
    o['application_data'] = {
        'klustaviewa': {
            'color': color,
        }
    }
    o['user_data'] = {}
    return o

    
# -----------------------------------------------------------------------------
# HDF5 helper functions
# -----------------------------------------------------------------------------
def create_kwx(path, channel_groups=None, nwavesamples=None, nfeatures=None,
               nchannels=None):
    """Create an empty KWX file.
    
    Arguments:
      * channel_groups: a dictionary 'ichannel_group': 'channel_group_info'
        where channel_group_info is a dictionary with the optional fields:
          * nchannels
          * nwavesamples
          * nfeatures
      * nwavesamples (common to all channel groups if set)
      * nfeatures (common to all channel groups if set)
      * nchannels (common to all channel groups if set)
    
    """
    file = tb.openFile(path, mode='w')
    file.createGroup('/', 'channel_groups')
    
    for ichannel_group, channel_group_info in sorted(iteritems(channel_groups)):
        nchannels_ = channel_group_info.get('nchannels', nchannels)
        nfeatures_ = channel_group_info.get('nfeatures', nfeatures)
        nwavesamples_ = channel_group_info.get('nwavesamples', nwavesamples)
        
        shank_path = '/channel_groups/channel_group{0:d}'.format(ichannel_group)
        
        # Create the HDF5 group for each channel group.
        file.createGroup('/channel_groups', 
                         'channel_group{0:d}'.format(ichannel_group))
                         
        # Create the tables.
        file.createTable(shank_path, 'spiketrain',
                         get_spiketrain_description())
        file.createTable(shank_path, 'spikesorting',
                         get_spikesorting_description(nfeatures=nfeatures_))
        file.createTable(shank_path, 'waveforms',
                         get_waveforms_description(nwavesamples=nwavesamples_,
                                                   nchannels=nchannels_))
                                                   
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
    file = tb.openFile(path, mode='w')
    
    for irecording, recording_info in sorted(iteritems(recordings)):
        nsamples_ = recording_info.get('nsamples', None)
        
        file.createGroup('/', 'recording{0:d}'.format(irecording))    
        recording_path = '/recording{0:d}'.format(irecording)
        
        file.createEArray(recording_path, 'data_{0:s}'.format(type), 
                          tb.Int16Atom(), 
                          (0, nchannels_tot), expectedrows=nsamples_)
    
    file.close()
   
def create_kwe(path, ):
    """Create an empty KWE file.
    
    Arguments:
      * 
    
    """
    file = tb.openFile(path, mode='w')
    
    # Create the tables.
    file.createTable('/', 'events',
                     get_events_description())
    file.createTable('/', 'event_types',
                     get_event_types_description())
                                                   
    file.close()
           
    