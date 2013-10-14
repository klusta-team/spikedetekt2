"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import tables as tb
import time
import shutil
from collections import OrderedDict

import numpy as np

from spikedetekt2.utils.six import iteritems


# -----------------------------------------------------------------------------
# Table descriptions
# -----------------------------------------------------------------------------
def get_spiketrain_description():
    return OrderedDict([
        ('sample', tb.Float64Col()),
        ('recording', tb.UInt16Col()),
        ('cluster', tb.UInt32Col()),
        ])
        
def get_spikesorting_description(nfeatures=None):
    return OrderedDict([
        ('features', tb.Float32Col(shape=(nfeatures,))),
        ('masks', tb.UInt8Col(shape=(nfeatures,))),
        ('cluster_original', tb.UInt32Col()),
        ])
    
def get_waveforms_description(nsamples=None, nchannels=None):
    return OrderedDict([
        ('waveform_filtered', tb.Int16Col(shape=(nsamples*nchannels))),
        ('waveform_raw', tb.Int16Col(shape=(nsamples*nchannels))),
        ])

def get_events_description():
    return OrderedDict([
        ('sample', tb.UInt64Col()),
        ('event_type', tb.UInt16Col()),
        ('recording', tb.UInt16Col()),
        ])


# -----------------------------------------------------------------------------
# HDF5 helper functions
# -----------------------------------------------------------------------------
def create_kwx(path, channel_groups=None, nsamples=None, nfeatures=None,
               nchannels=None):
    """Create an empty KWX file.
    
    Arguments:
      * channel_groups: a dictionary 'ichannel_group': 'channel_group_info'
        where channel_group_info is a dictionary with the optional fields:
          * nchannels
          * nsamples
          * nfeatures
      * nsamples (common to all channel groups if set)
      * nfeatures (common to all channel groups if set)
      * nchannels (common to all channel groups if set)
    
    """
    file = tb.openFile(path, mode='w')
    file.createGroup('/', 'channel_groups')
    
    for ichannel_group, channel_group_info in sorted(iteritems(channel_groups)):
        nchannels_ = channel_group_info.get('nchannels', nchannels)
        nfeatures_ = channel_group_info.get('nfeatures', nfeatures)
        nsamples_ = channel_group_info.get('nsamples', nsamples)
        
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
                         get_waveforms_description(nsamples=nsamples_,
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
        
        file.createEArray(recording_path, 'data_{0:d}'.format(type), 
                          tb.Int16Atom(), 
                          (0, nchannels_tot), expectedrows=nsamples_)
    
    file.close()
    
    