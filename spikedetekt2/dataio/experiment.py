"""Manage experiments."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import json

from spikedetekt2.dataio.kwik_creation import (create_kwik, 
    create_kwik_channel, create_kwik_cluster_group, create_kwik_cluster,
    create_kwik_recording, create_kwik_event_type,
    create_kwik_channel_group, create_kwik_main)
    
from spikedetekt2.utils.six import iteritems, string_types


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
    # TODO
    pass


# -----------------------------------------------------------------------------
# Experiment creation
# -----------------------------------------------------------------------------
def create_experiment(name=None, dir=None, filenames=None,
    channel_groups_info=None, event_types_info=None, recordings_info=None):
    """Create the JSON/HDF5 files forming an experiment.
    
    Arguments:
    
      * name: the experiment's name, which will be used to create the filenames
        if they aren't provided
      * dir: the directory where to save the files
      * filenames (optional): the dictionary with the filenames for each file
      * probe_info: info about the probe
      * events_info: info about the events
      * recordings_info: info about the recordings
    
    """
    if filenames is None:
        # TODO
        filenames = {}
        
    assert isinstance(name, string_types), ("You must specify an "
        "experiment's name.")
        
    # Get the filenames.
    path_kwik = filenames.get('kwik', None)
    path_kwx = filenames.get('kwx', None)
    paths_kwd = filenames.get('kwd', None)
    path_kwe = filenames.get('kwe', None)
    
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
    
    
    
