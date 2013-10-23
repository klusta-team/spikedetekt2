"""Manage experiments."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import json

import numpy as np
import pandas as pd

from selection import select
from spikedetekt2.dataio.kwik import get_filenames    
from spikedetekt2.utils.six import (iteritems, string_types, iterkeys, 
    itervalues)
from spikedetekt2.utils.wrap import wrap


# -----------------------------------------------------------------------------
# Experiment class
# -----------------------------------------------------------------------------
class Experiment(object):
    """An Experiment instance holds all information related to an
    experiment. One can access any information using a logical structure
    that is somewhat independent from the physical representation on disk.
    """
    def __init__(self, name=None, dir=None):
        # Read the
        exp = read_experiment(name=name, dir=dir)
        
class ChannelGroup(object):
    def __init__(self, name=None, graph=None, 
                 application_data=None, user_data=None,):
        pass
    

# -----------------------------------------------------------------------------
# Create experiment
# -----------------------------------------------------------------------------
def create_experiment(name=None, dir=None, filenames=None,):
    """Create the JSON/HDF5 files forming an experiment.
    
    TODO
    
    """
    if prm is None:
        prm = {}
    
    # Generate the filenames from the experiment's name if filenames are
    # not provided.
    if filenames is None:
        filenames = get_filenames(name)
        
    if dir is None:
        dir = os.path.abspath(os.getcwd())
        
    assert isinstance(name, string_types), ("You must specify an "
        "experiment's name.")
        
    # Get the filenames.
    path_kwik = os.path.join(dir, filenames.get('kwik', None))
    path_kwx = os.path.join(dir, filenames.get('kwx', None))
    paths_kwd = {key: os.path.join(dir, val) 
        for key, val in iteritems(filenames.get('kwd', {}))}
    

# -----------------------------------------------------------------------------
# Read experiment
# -----------------------------------------------------------------------------
def read_experiment(name=None, dir=None, filenames=None):
    if filenames is None:
        filenames = get_filenames(name)
        
    if dir is None:
        dir = os.path.abspath(os.getcwd())
        
    # Get the filenames.
    path_kwik = os.path.join(dir, filenames.get('kwik', None))
    path_kwx = os.path.join(dir, filenames.get('kwx', None))
    paths_kwd = {key: os.path.join(dir, val) 
        for key, val in iteritems(filenames.get('kwd', {}))}
    path_kwe = os.path.join(dir, filenames.get('kwe', None))
    
    # TODO: read the files and return an Experiment object.
