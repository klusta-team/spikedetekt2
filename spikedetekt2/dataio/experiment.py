"""Manage experiments."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.dataio.kwik_creation import (create_kwik, 
    create_kwik_channel, create_kwik_cluster_group, create_kwik_cluster,
    create_kwik_recording, create_kwik_event_type)
    
from spikedetekt2.utils.six import iteritems


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
    probe_info=None, events_info=None, recordings_info=None):
    """Create the JSON/HDF5 files forming an experiment.
    
    Arguments:
      * name: the experiment's name, which will be used to create the filenames
        if they aren't provided
      * dir: the directory where to save the files
      * filenames (optional): the dictionary with the filenames for each file
      * probe: the probe dictionary
    
    """
    if filenames is None:
        # TODO
        pass
        
    # Create the .kwik file.
    path_kwik = filenames['kwik']
    
    channel_groups = []
    # TODO
    # create_kwik_channel_group(
                # ichannel_group=0,
                # name='my channel group',
                # graph=[[0,1], [1, 2]],
                # channels=[
                    # create_kwik_channel(
                        # name='my first channel',
                        # ignored=False,
                        # position=[0., 0.],
                        # voltage_gain=10.,
                        # display_threshold=None),
                # ],
    
    # cluster_groups=[
                    # create_kwik_cluster_group(color=2, name='my cluster group',
                        # clusters=[
                             # create_kwik_cluster(color=4),
                        # ])
                # ],
    
    kwik = create_kwik(path_kwik,
        name=name,
        channel_groups=channel_groups,
        recordings=[  # TODO
            # create_kwik_recording(
                # irecording=0,
                # start_time=0.,
                # start_sample=0,
                # sample_rate=20000.,
                # band_low=100.,
                # band_high=3000.,
                # bit_depth=16),
        ],
        event_types=[],  # TODO
    )
    
    # TODO: create empty HDF5 files
    pass
    
    
