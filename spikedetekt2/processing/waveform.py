"""Alignment routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# Waveform class
# -----------------------------------------------------------------------------
class Waveform(object):
    def __init__(self, waveforms=None, masks=None, waveform_start=None, 
                 peak_rel=None, channel_group=None):
        self.waveforms = waveforms
        self.masks = masks
        self.s_start = waveform_start
        self.peak_abs = peak_rel + waveform_start
        self.channel_group = channel_group
        
    def __cmp__(self, other):
        return self.peak_abs - other.peak_abs


# -----------------------------------------------------------------------------
# Waveform extraction
# -----------------------------------------------------------------------------
def extract_waveform(component, chunk_strong=None, chunk_weak=None,
                     threshold_strong=None, threshold_weak=None, 
                     probe=None, **prm):
    pass
    # Find the channel_group of the spike
    # 
    
    
