"""Handle user-specified and default parameters."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.utils import get_pydict


# -----------------------------------------------------------------------------
# Python script <==> dictionaries conversion
# -----------------------------------------------------------------------------
def get_probe(filename=None, **kwargs):
    return get_pydict(filename=filename, **kwargs)

def get_adjacency_graph(prb):
    # TODO
    pass
    
# TODO: Probe class
class Probe(object):
    pass
