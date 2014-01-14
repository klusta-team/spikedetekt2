"""Main module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio import BaseRawDataReader, read_raw, excerpt_step
from spikedetekt2.processing import (bandpass_filter, apply_filter, 
    get_threshold, connected_components, extract_waveform,
    compute_pcs, project_pcs, DoubleThreshold)
from spikedetekt2.utils import (Probe, iterkeys, debug, info, warn, exception,
    display_params)


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
    waveforms = [extract_waveform(component,
                                  chunk_extract=chunk_extract,
                                  chunk_fil=chunk_fil,
                                  chunk_raw=chunk_raw,
                                  threshold_strong=threshold.strong,
                                  threshold_weak=threshold.weak,
                                  probe=probe,
                                  **prm) 
                 for component in components]
    # Remove skipped waveforms (in overlapping chunk sections).
    waveforms = [w for w in waveforms if w is not None]
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
            features = project_pcs(waveform, pcs)
            spikes.features_masks[i,:,0] = features.ravel()


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
    nchannels = prm.get('nchannels', None)
    
    # Ensure a RawDataReader is instantiated.
    if raw_data is not None:
        if not isinstance(raw_data, BaseRawDataReader):
            raw_data = read_raw(raw_data, nchannels=nchannels)
    else:
        raw_data = read_raw(experiment)
    # TODO: read from existing KWD file
    # TODO: when reading from .DAT file, convert into KWD at the same time
    # TODO: add_recording in Experiment as we go through the DAT files
    
    # Log.
    info("Starting process on {0:s}".format(str(raw_data)))
    debug("Parameters: \n" + (display_params(prm)))
    
    # Get the strong-pass filter.
    filter = bandpass_filter(**prm)
    
    # Compute the strong threshold across excerpts uniformly scattered across the
    # whole recording.
    threshold = get_threshold(raw_data, filter=filter, **prm)
    debug("Threshold: " + str(threshold))
    
    # Loop through all chunks with overlap.
    for chunk in raw_data.chunks(chunk_size=chunk_size, 
                                 chunk_overlap=chunk_overlap,):
        # Log.
        info("Processing chunk {0:s}...".format(chunk))
                                 
        # Filter the (full) chunk.
        chunk_raw = chunk.data_chunk_full  # shape: (nsamples, nchannels)
        chunk_fil = apply_filter(chunk_raw, filter=filter)
        
        # Apply thresholds.
        chunk_detect, chunk_threshold = apply_threshold(chunk_fil, 
            threshold=threshold, **prm)
        
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
        
    # Feature extraction.
    save_features(experiment, **prm)
    
    