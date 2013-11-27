"""Main module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio import BaseRawDataReader, read_raw, excerpt_step
from spikedetekt2.processing import (bandpass_filter, apply_filter, 
    get_threshold, apply_threshold, connected_components, extract_waveform,
    compute_pcs, project_pcs)
from spikedetekt2.utils import Probe, iterkeys


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
def add_waveform(experiment, waveform):
    experiment.channel_groups[waveform.channel_group].spikes.add(
        time_samples=waveform.s_offset, 
        time_fractional=waveform.s_frac_part,
        recording=waveform.recording,
        waveforms_raw=waveform.raw, 
        waveforms_filtered=waveform.fil,
        masks=waveform.masks,
    )
    
def save_features(experiment, nwaveforms_max=None, npcs=None):
    """Compute the features from the waveforms and save them in the experiment
    dataset."""
    for chgrp in iterkeys(experiment.channel_groups):
        spikes = experiment.channel_groups[chgrp].spikes
        # Extract a subset of the saveforms.
        nspikes = len(spikes)
        nwaveforms = min(nspikes, nwaveforms_max)
        step = excerpt_step(nspikes, nexcerpts=nwaveforms, excerpt_size=1)
        waveforms_subset = spikes.waveforms_filtered[::step]
        # Compute the PCs.
        pcs = compute_pcs(waveforms_subset, npcs=npcs)
        # Project the waveforms on the PCs and compute the features.
        # WARNING: optimization: we could load and project waveforms by chunks.
        for i, waveform in enumerate(spikes.waveforms_filtered):
            features = project_pcs(waveform, pcs)
            spikes.features_masks[i,:,0] = features.ravel()
            # TODO: add masks
        
    
# -----------------------------------------------------------------------------
# Main loop
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
        # For now, we use the same binary chunk for detection and extraction
        # +/-chunk, or abs(chunk), depending on the parameter 'detect_spikes'.
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
        # Remove skipped waveforms (in overlapping chunk sections).
        waveforms = [w for w in waveforms if w is not None]
                        
        # We sort waveforms by increasing order of fractional time.
        for waveform in sorted(waveforms):
            add_waveform(experiment, waveform)
            
    # Feature extraction.
    save_features(experiment, 
                  nwaveforms_max=prm['pca_nwaveforms_max'],
                  npcs=prm['nfeatures_per_channel'])
    


