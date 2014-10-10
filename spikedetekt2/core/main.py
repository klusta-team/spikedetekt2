"""Main module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging

import numpy as np
import tables as tb

from .progressbar import ProgressReporter
from kwiklib.dataio import (BaseRawDataReader, read_raw, excerpt_step,
    to_contiguous, convert_dtype, KwdRawDataReader)
from spikedetekt2.processing import (bandpass_filter, apply_filter,get_noise_cov,get_whitening_matrix,get_whitening_matrix_cholesky,get_whitening_matrix_scipy,whiten, decimate,
    get_threshold, connected_components, extract_waveform,
    compute_pcs, project_pcs, DoubleThreshold,plot_diagnostics_twothresholds)
from kwiklib.utils import (Probe, iterkeys, debug, info, warn, exception,
    display_params, FileLogger, register, unregister)

from IPython import embed 
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

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
    
def save_features(experiment, whiteningmat, **prm):
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
            prefeatures = project_pcs(waveform, pcs)
            if prm['whiten']:
                features = np.dot( whiteningmat,prefeatures)
            else:
	        features = prefeatures
            #embed()
            # featur
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
    
    # Compute the strong threshold across excerpts uniformly scattered across the
    # whole recording.
    threshold = get_threshold(raw_data, filter=filter, 
                              channels=probe.channels, **prm)
    assert not np.isnan(threshold.weak)
    assert not np.isnan(threshold.strong)
    debug("Threshold: " + str(threshold))
    
    # Progress bar.
    progress_bar = ProgressReporter(period=30.)
    nspikes = 0
    
    # Loop through all chunks with overlap.
    for chunknum, chunk in enumerate(raw_data.chunks(chunk_size=chunk_size, 
                                 chunk_overlap=chunk_overlap,)):
        # Log.
        print 'chunknum is ', chunknum
        # embed()
        debug("Processing chunk {0:s}...".format(chunk))
        
        nsamples = chunk.nsamples
        rec = chunk.recording
        nrecs = chunk.nrecordings
        s_end = chunk.s_end
                                 
        # Filter the (full) chunk.
        chunk_raw = chunk.data_chunk_full  # shape: (nsamples, nchannels)
        chunk_fil = apply_filter(chunk_raw, filter=filter)
        
        i = chunk.keep_start - chunk.s_start
        j = chunk.keep_end - chunk.s_start
            
        # Add the data to the KWD files.
        if prm.get('save_raw', False):
            # Do not append the raw data to the .kwd file if we're already reading
            # from the .kwd file.
            if not isinstance(raw_data, KwdRawDataReader):
                # Save raw data.
                experiment.recordings[chunk.recording].raw.append(convert_dtype(chunk.data_chunk_keep, np.int16))
            
        if prm.get('save_high', False):
            # Save high-pass filtered data: need to remove the overlapping
            # sections.
            chunk_fil_keep = chunk_fil[i:j,:]
            experiment.recordings[chunk.recording].high.append(convert_dtype(chunk_fil_keep, np.int16))
            
        if prm.get('save_low', True):
            # Save LFP.
            chunk_low = decimate(chunk_raw)
            chunk_low_keep = chunk_low[i//16:j//16,:]
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
        #embed()
        # Log number of spikes in the chunk.
        nspikes += len(waveforms)
        
        ##noisecov = get_noise_cov(chunk_fil,components)
        #if chunknum ==0: 
	regu = 0
	regcove = 0.1
	#whiteningmat, noisecov = get_whitening_matrix(chunk_fil,components,epsilon_fudge=1)
	#whiteningmat, noisecov = get_whitening_matrix_cholesky(chunk_fil,components,reg = regu, regcov = regcove)
	whiteningmat, noisecov = get_whitening_matrix_scipy(chunk_fil,components)
	chunk_whitened_fildata = whiten(chunk_fil, whiteningmat)
	whitened_cov = np.cov(chunk_whitened_fildata, rowvar =0)
    
	##noisecov = get_noise_cov(chunk_fil,components)
	#whiteningmat_raw, noisecov_raw = get_whitening_matrix(chunk_raw,components,epsilon_fudge=1)
	#whiteningmat_raw, noisecov_raw = get_whitening_matrix_cholesky(chunk_raw,components,reg = regu, regcov = regcove)
	whiteningmat_raw, noisecov_raw = get_whitening_matrix_scipy(chunk_raw,components)
	chunk_whitened_rawdata = whiten(chunk_raw, whiteningmat_raw)
	whitened_cov_raw = np.cov(chunk_whitened_rawdata, rowvar =0)
    
    
	total_height = 3
	total_width = 2
	#print 'Yo, I got to line 129 of debug_manual.py'
	gs = gridspec.GridSpec(total_height,total_width)
	fig2 = plt.figure()
	noiseaxis = fig2.add_subplot(gs[0,0:total_width])
	noiseaxis.set_title('Noise covariance matrix',fontsize=10)
	imnoise = noiseaxis.imshow(noisecov,interpolation="nearest",aspect="auto")
	plt.colorbar(imnoise)
	whitenedcovaxis = fig2.add_subplot(gs[1,0:total_width])
	whitenedcovaxis.set_title('Whitened covariance matrix',fontsize=10)
	imwhitecov = whitenedcovaxis.imshow(whitened_cov,interpolation="nearest",aspect="auto")
	plt.colorbar(imwhitecov)
	whiteningmataxis = fig2.add_subplot(gs[2,0:total_width])
	whiteningmataxis.set_title('Whitening matrix',fontsize=10)
	imwhitenmat = whiteningmataxis.imshow(whiteningmat,interpolation="nearest",aspect="auto")
	plt.colorbar(imwhitenmat)
	fig2.savefig('covs_%d_scipy.pdf' %(nspikes))
	#fig2.savefig('covs_%d_reg_%d_regcov_%d.pdf' %(nspikes,regu,regcove))
	
	gs = gridspec.GridSpec(total_height,total_width)
	fig3 = plt.figure()
	noiseaxis = fig3.add_subplot(gs[0,0:total_width])
	noiseaxis.set_title('Raw Noise covariance matrix',fontsize=10)
	imnoise = noiseaxis.imshow(noisecov_raw,interpolation="nearest",aspect="auto")
	plt.colorbar(imnoise)
	whitenedcovaxis = fig3.add_subplot(gs[1,0:total_width])
	whitenedcovaxis.set_title('Raw Whitened covariance matrix',fontsize=10)
	imwhitecov = whitenedcovaxis.imshow(whitened_cov_raw,interpolation="nearest",aspect="auto")
	plt.colorbar(imwhitecov)
	whiteningmataxis = fig3.add_subplot(gs[2,0:total_width])
	whiteningmataxis.set_title('Raw Whitening matrix',fontsize=10)
	imwhitenmat = whiteningmataxis.imshow(whiteningmat_raw,interpolation="nearest",aspect="auto")
	plt.colorbar(imwhitenmat)
	fig3.savefig('rawcovs_%d_scipy.pdf' %(nspikes))
        #fig3.savefig('rawcovs_%d_reg_%d_regcov_%d.pdf' %(nspikes,regu,regcove))
        #embed()
        #embed()
        # embed()
        #If using debug module
        if prm['debug'] == True:
            print 'debugging'
            plot_diagnostics_twothresholds(threshold = threshold,probe = probe,components = components,chunk = chunk, chunk_detect= chunk_detect,chunk_threshold= chunk_threshold, chunk_fil=chunk_fil, chunk_white = chunk_whitened_fildata,chunk_white_raw = chunk_whitened_rawdata, chunk_raw=chunk_raw, reg = regu, regcov = regcove, **prm)
        
        if chunknum ==0: 
	    globalwhiteningmat = whiteningmat
        #embed()
        # We sort waveforms by increasing order of fractional time.
        [add_waveform(experiment, waveform) for waveform in sorted(waveforms)]
        
        # Update the progress bar.
        progress_bar.update(rec/float(nrecs) + (float(s_end) / (nsamples*nrecs)),
            '%d spikes found.' % (nspikes))
        
        # DEBUG: keep only the first shank.
        if _debug:
            break
        
    # Feature extraction.
    save_features(experiment, globalwhiteningmat, **prm)
    
    close_file_logger(LOGGER_FILE)
    progress_bar.finish()
    