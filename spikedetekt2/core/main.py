"""Main module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio import BaseRawDataReader, read_raw
from spikedetekt2.processing import bandpass_filter, apply_filter


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------
def run(raw_data=None, experiment=None, prm=None, prb=None, **kwargs):
    """This main function takes raw data (either as a RawReader, or a path
    to a filename, or an array) and executes the main algorithm (filtering, 
    spike detection, extraction...)."""
    
    assert experiment is not None, ("An Experiment instance needs to be "
        "provided in order to write the output.")
    
    chunk_size = prm.get('chunk_size', None)
    chunk_overlap = prm.get('chunk_overlap', 0)
    sample_rate = prm['sample_rate']
    
    if raw_data is not None:
        if not isinstance(raw_data, BaseRawDataReader):
            raw_data = read_raw(raw_data, 
                                chunk_size=chunk_size, 
                                chunk_overlap=chunk_overlap, 
                                **kwargs)
    else:
        raw_data = read_raw(experiment, 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap, 
                            **kwargs)
        
    # TODO: from PRM, create a RawDataReader that automatically concatenates
    # the different dat files.
    # Now, we can assume that raw_data is a valid RawDataReader instance.
    
    # Filtering.
    # F_LOW = 500. # low pass frequency (Hz)
    # F_HIGH_FACTOR = 0.95 # high pass frequency as a proportion of the Nyquist freq, used to derive F_HIGH, i.e. F_HIGH = 0.95*SAMPLERATE/2 here
    # BUTTER_ORDER = 3 # Order of butterworth filter
    filter = bandpass_filter(order=prm['filter_butter_order'],
                             rate=sample_rate,
                             low=prm['filter_low'],
                             high=prm.get('filter_high',
                                          (prm['filter_high_factor'] * 
                                                sample_rate * .5)),
                             )
    
    # Get the threshold: 50 chunks of 1s evenly scattered along the recording
    # threshold = std (1 for all channels for now, but may be changed later)
    # TODO: raw_data.next_excerpt()
    
    for chunk in raw_data:
        pass
        # Find connected component (high threshold for seeds).
        # For each component
            # Alignment.
            # t_ = (\sum_{ct} (y_{ct}-B)+^p t) / (\sum_{ct} (y_{ct}-B)+^p)
            # f = interp1d on all channels
                # 1: not upsample !!!, find t_, and interp1d()(t_::10)
                # ((2: upsample: int(t_)::10))
                # (3: upsample t_::10)

            # other option:
                # upsample
                # peak on each channel
                # weight as above
    
            # Masking. 
            # linear interp between 2 thresholds for the max sample on each channel
            
    # Feature extraction.
        # PCA: sample 10000 waveforms evenly in time
        # specify the total number of waveforms
        
    
    


