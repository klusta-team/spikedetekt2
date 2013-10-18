"""Manage experiments."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import json

import pandas as pd

from spikedetekt2.dataio.kwik_creation import (create_kwik, 
    create_kwik_channel, create_kwik_cluster_group, create_kwik_cluster,
    create_kwik_recording, create_kwik_event_type,
    create_kwik_channel_group, create_kwik_main, create_kwx, create_kwd,
    create_kwe)
from spikedetekt2.dataio.files import generate_filenames    
from spikedetekt2.utils.six import iteritems, string_types
from spikedetekt2.utils.wrap import wrap


# -----------------------------------------------------------------------------
# Experiment class
# -----------------------------------------------------------------------------
class Experiment(object):
    """An Experiment instance holds all information related to an
    experiment. One can access any information using a logical structure
    that is somewhat independent from the physical representation on disk.
    
    This class is read-only. It does not define any method modifying
    the data. To update the data, one needs to use one of the existing
    update functions.
    """
    def __init__(self, name=None, dir=None):
        pass
        
class Channels(object):
    def __init__(self, **kwargs):
        pass
    


# -----------------------------------------------------------------------------
# Experiment creation
# -----------------------------------------------------------------------------
def create_experiment(name=None, dir=None, filenames=None, nchannels_tot=None,
    channel_groups_info=None, event_types_info=None, recordings_info=None,
    spikedetekt_params=None):
    """Create the JSON/HDF5 files forming an experiment.
    
    Arguments:
    
      * name: the experiment's name, which will be used to create the filenames
        if they aren't provided
      * dir: the directory where to save the files
      * filenames (optional): the dictionary with the filenames for each file
      * channel_groups_info: info about the probe and channel groups
      * event_types_info: info about the event types
      * recordings_info: info about the recordings
    
    """
    if spikedetekt_params is None:
        spikedetekt_params = {}
    
    # Generate the filenames from the experiment's name if filenames are
    # not provided.
    if filenames is None:
        filenames = generate_filenames(name)
        
    if dir is None:
        dir = os.path.abspath(os.getcwd())
        
    assert isinstance(name, string_types), ("You must specify an "
        "experiment's name.")
        
    # Get the number of non-ignored channels per channel group (this is
    # used as information when creating the waveforms table in HDF5).
    nchannels_list = [len([ch for ch in channel_group_info['channels']
                              if not ch.get('ignored', False)]) 
        for channel_group_info in channel_groups_info]
        
    # Get the filenames.
    path_kwik = os.path.join(dir, filenames.get('kwik', None))
    path_kwx = os.path.join(dir, filenames.get('kwx', None))
    paths_kwd = {key: os.path.join(dir, val) 
        for key, val in iteritems(filenames.get('kwd', {}))}
    path_kwe = os.path.join(dir, filenames.get('kwe', None))
    
    # Channel groups.
    channel_groups = [
        create_kwik_channel_group(
            ichannel_group=ichannel_group,
            channels=[
                create_kwik_channel(**channel_info)
                    for ichannel, channel_info in enumerate(
                            channel_group_info.pop('channels', []))
            ],
            cluster_groups=[
                create_kwik_cluster_group(
                    clusters=[
                        create_kwik_cluster(**cluster_info)
                            for icluster, cluster_info in enumerate(
                                cluster_group_info.pop('clusters', [])
                                )
                    ],
                    **cluster_group_info)
                    for icluster_group_info, cluster_group_info in enumerate(
                            channel_group_info.pop('cluster_groups', []))
            ],
            **channel_group_info
            )
        for ichannel_group, channel_group_info in enumerate(channel_groups_info)
    ]
    
    # Recordings
    recordings = [
        create_kwik_recording(irecording=irecording, **recording_info)
            for irecording, recording_info in enumerate(recordings_info)
    ]
    
    # Event types
    event_types = [
        create_kwik_event_type(**event_type_info)
            for ievent_type, event_type_info in enumerate(event_types_info)
    ]
    
    # Create the KWIK dict.
    kwik = create_kwik_main(
        name=name,
        channel_groups=channel_groups,
        recordings=recordings,
        event_types=event_types, 
    )
    # Save the dict in the KWIK file.
    if path_kwik:
        create_kwik(path_kwik, kwik=kwik)
    
    # Create the KWX file.
    channel_groups_kwx = [
        {
            'nchannels': nchannels,
            'nfeatures': nchannels * spikedetekt_params.get('fetdim', 0),
            'nwavesamples': spikedetekt_params.get('nwavesamples', 0),
        }
        for nchannels in nchannels_list
    ]
    if path_kwx:
        create_kwx(path_kwx, 
                   channel_groups=channel_groups_kwx)
    
    # Create the KWD files.
    for type in ('raw', 'high', 'low'):
        path_kwd = paths_kwd.get(type, None)
        if path_kwd:
            # TODO: recordings kwarg with 'nsamples' (expected) for each recording
            create_kwd(path_kwd, type=type, nchannels_tot=nchannels_tot)

    # Create the KWE file.
    if path_kwe:
        create_kwe(path_kwe)
        