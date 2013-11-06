"""Filtering tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2 import (read_raw, get_threshold, bandpass_filter)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_thresholding_1():
    duration = 10.
    sample_rate = 20000.
    filter_low = 200.
    filter_high = 1000.
    nchannels = 5
    nsamples = int(duration * sample_rate)
    nexcerpts = 3
    excerpt_size = 10000
    
    X = np.random.randn(nsamples, nchannels)
    raw_data = read_raw(X)
    
    filter = bandpass_filter(order=3,
                             rate=sample_rate,
                             low=filter_low,
                             high=filter_high,
                             )
                             
    threshold = get_threshold(raw_data, filter=filter, 
                              nexcerpts=nexcerpts,
                              excerpt_size=excerpt_size,
                              threshold_std_factor=4.5)
    
    assert np.abs(threshold - 4.5) < .5
    
    