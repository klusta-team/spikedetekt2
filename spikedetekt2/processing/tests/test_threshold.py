"""Filtering tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from nose.tools import assert_almost_equal as almeq

from spikedetekt2 import (read_raw, get_threshold, bandpass_filter,)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_get_threshold_1():
    duration = 10.
    sample_rate = 20000.
    filter_low = 10.
    filter_high = 10000.
    nchannels = 5
    nsamples = int(duration * sample_rate)
    nexcerpts = 3
    excerpt_size = 10000
    
    X = np.random.randn(nsamples, nchannels)
    raw_data = read_raw(X)
    
    filter = bandpass_filter(filter_butter_order=3,
                             sample_rate=sample_rate,
                             filter_low=filter_low,
                             filter_high=filter_high,
                             )
                             
    threshold = get_threshold(raw_data, filter=filter, 
                              nexcerpts=nexcerpts,
                              excerpt_size=excerpt_size,
                              threshold_std_factor=4.5)

    assert np.abs(np.array(threshold) - 4.5) < .5
    
    