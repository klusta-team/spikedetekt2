"""Alignment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.processing import extract_waveform
from spikedetekt2.utils import Probe


# -----------------------------------------------------------------------------
# Test probe
# -----------------------------------------------------------------------------
n = 5
probe_adjacency_list = [(i, i+1) for i in range(n-1)]
PROBE = Probe({'channel_groups': [{'channels': range(n),
                                   'graph': probe_adjacency_list}]})


# -----------------------------------------------------------------------------
# Test component
# -----------------------------------------------------------------------------
CHUNK_WEAK = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ]
CHUNK_STRONG = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
COMPONENT = [(1, 2), (2, 2), (2, 3), 
             (3, 3), (4, 3), (4, 4)]
JOIN_SIZE = 1 
THRESHOLD_STRONG = .8
THRESHOLD_WEAK = .5


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_extract_waveform_1():
    extract_waveform(COMPONENT,
                     chunk_strong=CHUNK_STRONG,
                     chunk_weak=CHUNK_WEAK,
                     threshold_strong=THRESHOLD_STRONG,
                     threshold_weak=THRESHOLD_WEAK, 
                     probe=PROBE)
    
    
    