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
        # shape: (nsamples, nchannels)
        chunk_raw = chunk.data_chunk_full
        chunk_fil = apply_filter(chunk_raw, filter=filter)
        
        # Apply thresholds.
        if prm['detect_spikes'] == 'positive':
            chunk_detect = chunk_fil
        elif prm['detect_spikes'] == 'negative':
            chunk_detect = -chunk_fil
        elif prm['detect_spikes'] == 'both':
            chunk_detect = np.abs(chunk_fil)
        chunk_strong = chunk_detect > threshold_strong  # shape: (nsamples, nchannels)
        chunk_weak = chunk_detect > threshold_weak
        
        # Find connected component (strong threshold). Return list of
        # Component instances.
        components = connected_components(chunk_strong=chunk_strong, 
                                          chunk_weak=chunk_weak,
                                          probe_adjacency_list=probe.adjacency_list,
                                          chunk=chunk,
                                          **prm)
        
        # Now we extract the spike in each component.
        chunk_extract = chunk_detect  # shape: (nsamples, nchannels)
        # This is a list of Waveform instances.
        waveforms = [extract_waveform(component,
                                      chunk_extract=chunk_extract,
                                      chunk_fil=chunk_fil,
                                      chunk_raw=chunk_raw,
                                      threshold_strong=threshold_strong,
                                      threshold_weak=threshold_weak,
                                      probe=probe,
                                      **prm) 
                     for component in components]
                        
        # We sort waveforms by increasing order of fractional time.
        for waveform in sorted(waveforms):
            # experiment.channel_groups[0].spikes.add(#TODO
                                                    # )
            
            # TODO
            # s_offset = s_start+s_peak
                    # sf_offset = s_start + sf_peak
                    # if keep_start<=s_offset<keep_end:
                        # spike_count += 1
            
            print(waveform)
            
    # Feature extraction.
        # PCA: sample 10000 waveforms evenly in time
        # specify the total number of waveforms
        
    
    


