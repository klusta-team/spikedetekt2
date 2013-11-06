"""Main module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio import BaseRawDataReader, read_raw
from spikedetekt2.processing import (bandpass_filter, apply_filter, 
    get_threshold)


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
def run(raw_data=None, experiment=None, prm=None, prb=None):
    """This main function takes raw data (either as a RawReader, or a path
    to a filename, or an array) and executes the main algorithm (filtering, 
    spike detection, extraction...)."""
    
    assert experiment is not None, ("An Experiment instance needs to be "
        "provided in order to write the output.")
    
    # Get parameters from the PRM dictionary.
    sample_rate = prm['sample_rate']
    chunk_size = prm.get('chunk_size', None)
    chunk_overlap = prm.get('chunk_overlap', 0)
    nexcerpts = prm['nexcerpts']
    excerpt_size = prm['excerpt_size']
    filter_butter_order = prm['filter_butter_order']
    filter_high = prm['filter_high']
    filter_low = prm['filter_low']
    threshold_high_std_factor = prm['threshold_high_std_factor']
    threshold_low_std_factor = prm['threshold_low_std_factor']
    
    # Ensure a RawDataReader is instanciated.
    # TODO: concatenate DAT files
    if raw_data is not None:
        if not isinstance(raw_data, BaseRawDataReader):
            raw_data = read_raw(raw_data)
    else:
        raw_data = read_raw(experiment)
    
    # Get the high-pass filter.
    filter = bandpass_filter(order=filter_butter_order,
                             rate=sample_rate,
                             low=filter_low,
                             high=filter_high,)
    
    # Compute the high threshold across excerpts uniformly scattered across the
    # whole recording.
    factors = (threshold_high_std_factor, threshold_low_std_factor)
    threshold_high, threshold_low = get_threshold(raw_data, 
                                        filter=filter, 
                                        nexcerpts=nexcerpts,
                                        excerpt_size=excerpt_size,
                                        threshold_std_factor=factors)
    
    # Loop through all chunks with overlap.
    for chunk in raw_data.chunks(chunk_size=chunk_size, 
                                 chunk_overlap=chunk_overlap,):
        # Filter the chunk.
        chunk_fil = apply_filter(chunk.data_chunk_keep, filter=filter)
        
        # Find connected component (high threshold for seeds).
        # For each component
            # Alignment.
            # Masking. 
            # linear interp between 2 thresholds for the max sample on each channel
            
    # Feature extraction.
        # PCA: sample 10000 waveforms evenly in time
        # specify the total number of waveforms
        
    
    


