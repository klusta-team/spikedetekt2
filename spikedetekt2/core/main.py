"""Main module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio import BaseRawDataReader, read_raw
from spikedetekt2.processing import (bandpass_filter, apply_filter, 
    get_threshold, apply_threshold, connected_components, extract_waveform)
from spikedetekt2.utils import Probe


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
def run(raw_data=None, experiment=None, prm=None, probe=None):
    """This main function takes raw data (either as a RawReader, or a path
    to a filename, or an array) and executes the main algorithm (filtering, 
    spike detection, extraction...)."""
    
    assert experiment is not None, ("An Experiment instance needs to be "
        "provided in order to write the output.")
    
    # Get parameters from the PRM dictionary.
    chunk_size = prm.get('chunk_size', None)
    chunk_overlap = prm.get('chunk_overlap', 0)
    
    # Ensure a RawDataReader is instanciated.
    # TODO: concatenate DAT files
    if raw_data is not None:
        if not isinstance(raw_data, BaseRawDataReader):
            raw_data = read_raw(raw_data)
    else:
        raw_data = read_raw(experiment)
    
    # Get the strong-pass filter.
    filter = bandpass_filter(**prm)
    
    # Compute the strong threshold across excerpts uniformly scattered across the
    # whole recording.
    threshold_strong, threshold_weak = get_threshold(raw_data, 
                                                     filter=filter, 
                                                     **prm)
    
    # Loop through all chunks with overlap.
    for chunk in raw_data.chunks(chunk_size=chunk_size, 
                                 chunk_overlap=chunk_overlap,):
        # Filter the (full) chunk.
        chunk_fil = apply_filter(chunk.data_chunk_full, filter=filter)
        
        # Apply strong threshold.
        chunk_strong = apply_threshold(chunk_fil, -threshold_strong,
                                       side='below')
        chunk_weak = apply_threshold(chunk_fil, -threshold_weak,
                                       side='below')
        
        # Find connected component (strong threshold).
        components = connected_components(chunk_strong=chunk_strong, 
                                          chunk_weak=chunk_weak,
                                          probe_adjacency_list=probe.adjacency_list,
                                          **prm)
        
        # For each component
        # TODO: waveform class with __cmp__ as a function of fractional peak
        waveforms = [extract_waveform(component,
                                      chunk_fil=chunk_fil,
                                      chunk_strong=chunk_strong,
                                      chunk_weak=chunk_weak,
                                      threshold_strong=threshold_strong,
                                      threshold_weak=threshold_weak,
                                      probe=probe,
                                      **prm) 
                     for component in components]
        # Sort waveforms by increasing order of fractional time.
        for waveform in sorted(waveforms):
            # TODO: channel group index? get the shank from the connected 
            # component, maybe a field of waveforms
            # experiment.channel_groups[0].spikes.add(#TODO
                                                    # )
            pass
            
    # Feature extraction.
        # PCA: sample 10000 waveforms evenly in time
        # specify the total number of waveforms
        
    
    


