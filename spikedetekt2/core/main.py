"""Main module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging

import numpy as np
import tables as tb

from kwiklib.dataio import (BaseRawDataReader, read_raw, excerpt_step,
    to_contiguous, convert_dtype)
from spikedetekt2.processing import (bandpass_filter, apply_filter, 
    get_threshold, connected_components, extract_waveform,
    compute_pcs, project_pcs, DoubleThreshold)
from kwiklib.utils import (Probe, iterkeys, debug, info, warn, exception,
    display_params, FileLogger, register, unregister)


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
def apply_threshold(chunk_fil, threshold=None, **prm):
    # Determine the chunk used for thresholding.
    if prm['detect_spikes'] == 'positive':
        chunk_detect = chunk_fil
    elif prm['detect_spikes'] == 'negative':
        chunk_detect = -chunk_fil
    elif prm['detect_spikes'] == 'both':
        chunk_detect = np.abs(chunk_fil)
    
    # Perform thresholding.
    # shape: (nsamples, nchannels)
    chunk_threshold = DoubleThreshold(
        strong=chunk_detect > threshold.strong,
        weak=chunk_detect > threshold.weak,
    )
    return chunk_detect, chunk_threshold

def extract_waveforms(chunk_detect=None, threshold=None,
                      chunk_fil=None, chunk_raw=None,
                      probe=None, components=None,
                      **prm):
    # For now, we use the same binary chunk for detection and extraction
    # +/-chunk, or abs(chunk), depending on the parameter 'detect_spikes'.
    chunk_extract = chunk_detect  # shape: (nsamples, nchannels)
    # This is a list of Waveform instances.
    waveforms = []
    for component in components:
        w = extract_waveform(component,
                             chunk_extract=chunk_extract,
                             chunk_fil=chunk_fil,
                             chunk_raw=chunk_raw,
                             threshold_strong=threshold.strong,
                             threshold_weak=threshold.weak,
                             probe=probe,
                             **prm)
        if w is not None:
            waveforms.append(w)
            
    # Remove skipped waveforms (in overlapping chunk sections).
    # waveforms = [w for w in waveforms if w is not None]
    return waveforms
    
def add_waveform(experiment, waveform, **prm):
    """Add a Waveform instance to an Experiment."""
    experiment.channel_groups[waveform.channel_group].spikes.add(
        time_samples=waveform.s_offset, 
        time_fractional=waveform.s_frac_part,
        recording=waveform.recording,
        waveforms_raw=waveform.raw, 
        waveforms_filtered=waveform.fil,
        masks=waveform.masks,
    )
    
def save_features(experiment, **prm):
    """Compute the features from the waveforms and save them in the experiment
    dataset."""
    nwaveforms_max = prm['pca_nwaveforms_max']
    npcs = prm['nfeatures_per_channel']
    
    for chgrp in iterkeys(experiment.channel_groups):
        spikes = experiment.channel_groups[chgrp].spikes
        # Extract a subset of the saveforms.
        nspikes = len(spikes)
        
        # We convert the extendable features_masks array to a 
        # contiguous array.
        if prm.get('features_contiguous', True):
            # Make sure to update the PyTables node after the recreation,
            # to avoid ClosedNodeError.
            spikes.features_masks = to_contiguous(spikes.features_masks, nspikes=nspikes)
        else:
            warn(("The features array has not been converted to a contiguous "
                  "array."))
        
        # Skip the channel group if there are no spikes.
        if nspikes == 0:
            continue
        nwaveforms = min(nspikes, nwaveforms_max)
        step = excerpt_step(nspikes, nexcerpts=nwaveforms, excerpt_size=1)
        waveforms_subset = spikes.waveforms_filtered[::step]
        # Compute the PCs.
        pcs = compute_pcs(waveforms_subset, npcs=npcs)
        # Project the waveforms on the PCs and compute the features.
        # WARNING: optimization: we could load and project waveforms by chunks.
        for i, waveform in enumerate(spikes.waveforms_filtered):
            # Convert waveforms from int16 to float32 with scaling
            # before computing PCA so as to avoid getting huge numbers.
            waveform = convert_dtype(waveform, np.float32)
            features = project_pcs(waveform, pcs)
            spikes.features_masks[i,:,0] = features.ravel()
    
    
# -----------------------------------------------------------------------------
# File logger
# -----------------------------------------------------------------------------
def create_file_logger(filename):
    # global LOGGER_FILE
    LOGGER_FILE = FileLogger(filename, name='file', 
        level=logging.DEBUG)
    register(LOGGER_FILE)
    return LOGGER_FILE

def close_file_logger(LOGGER_FILE):
    unregister(LOGGER_FILE)
  

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def run(raw_data=None, experiment=None, prm=None, probe=None, 
        _debug=False):
    """This main function takes raw data (either as a RawReader, or a path
    to a filename, or an array) and executes the main algorithm (filtering, 
    spike detection, extraction...)."""
    
    assert experiment is not None, ("An Experiment instance needs to be "
        "provided in order to write the output.")
    
    # Create file logger for the experiment.
    LOGGER_FILE = create_file_logger(experiment.gen_filename('log'))
    
    # Get parameters from the PRM dictionary.
    chunk_size = prm.get('chunk_size', None)
    chunk_overlap = prm.get('chunk_overlap', 0)
    nchannels = prm.get('nchannels', None)
    
    # Ensure a RawDataReader is instantiated.
    if raw_data is not None:
        if not isinstance(raw_data, BaseRawDataReader):
            raw_data = read_raw(raw_data, nchannels=nchannels)
    else:
        raw_data = read_raw(experiment)
    
    # Log.
    info("Starting process on {0:s}".format(str(raw_data)))
    debug("Parameters: \n" + (display_params(prm)))
    
    # Get the bandpass filter.
    filter = bandpass_filter(**prm)
    
    # Get the LFP filter.
    filter_low = bandpass_filter(sample_rate=prm['sample_rate'],
                                 filter_butter_order=prm['filter_butter_order'],
                                 filter_low=prm['filter_lfp_low'],
                                 filter_high=prm['filter_lfp_high'])
    
    # Compute the strong threshold across excerpts uniformly scattered across the
    # whole recording.
    threshold = get_threshold(raw_data, filter=filter, 
                              channels=probe.channels, **prm)
    debug("Threshold: " + str(threshold))
    
    # Loop through all chunks with overlap.
    for chunk in raw_data.chunks(chunk_size=chunk_size, 
                                 chunk_overlap=chunk_overlap,):
        # Log.
        info("Processing chunk {0:s}...".format(chunk))
                                 
        # Filter the (full) chunk.
        chunk_raw = chunk.data_chunk_full  # shape: (nsamples, nchannels)
        chunk_fil = apply_filter(chunk_raw, filter=filter)
        
        i = chunk.keep_start - chunk.s_start
        j = chunk.keep_end - chunk.s_start
            
        # Add the data to the KWD files.
        if prm.get('save_raw', False):
            # Save raw data.
            experiment.recordings[chunk.recording].raw.append(convert_dtype(chunk.data_chunk_keep, np.int16))
            
        if prm.get('save_high', False):
            # Save high-pass filtered data: need to remove the overlapping
            # sections.
            chunk_fil_keep = chunk_fil[i:j,:]
            experiment.recordings[chunk.recording].high.append(convert_dtype(chunk_fil_keep, np.int16))
            
        if prm.get('save_low', True):
            # Save LFP.
            chunk_low = apply_filter(chunk_raw, filter=filter_low)
            chunk_low_keep = chunk_low[i:j,:]
            experiment.recordings[chunk.recording].low.append(convert_dtype(chunk_low_keep, np.int16))
        
        # Apply thresholds.
        chunk_detect, chunk_threshold = apply_threshold(chunk_fil, 
            threshold=threshold, **prm)
        
        # Remove dead channels.
        dead = np.setdiff1d(np.arange(nchannels), probe.channels)
        chunk_detect[:,dead] = 0
        chunk_threshold.strong[:,dead] = 0
        chunk_threshold.weak[:,dead] = 0
        
        # Find connected component (strong threshold). Return list of
        # Component instances.
        components = connected_components(
            chunk_strong=chunk_threshold.strong, 
            chunk_weak=chunk_threshold.weak, 
            probe_adjacency_list=probe.adjacency_list,
            chunk=chunk, **prm)
        
        # Now we extract the spike in each component.
        waveforms = extract_waveforms(chunk_detect=chunk_detect,
            threshold=threshold, chunk_fil=chunk_fil, chunk_raw=chunk_raw, 
            probe=probe, components=components, **prm)
        
        # Log number of spikes in the chunk.
        info("Found {0:d} spikes".format(len(waveforms)))
        
        # We sort waveforms by increasing order of fractional time.
        [add_waveform(experiment, waveform) for waveform in sorted(waveforms)]
        
        # DEBUG: keep only the first shank.
        if _debug:
            break
        
    # Feature extraction.
    save_features(experiment, **prm)
    
    close_file_logger(LOGGER_FILE)
    