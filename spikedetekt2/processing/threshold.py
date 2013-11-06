"""Thresholding routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal

from spikedetekt2.processing import apply_filter


# -----------------------------------------------------------------------------
# Thresholding
# -----------------------------------------------------------------------------
def _n_sum_sum2(x):
    return np.vstack(((x.shape[0],) * x.shape[1],
                      np.sum(x, axis=0),
                      np.sum(x * x, axis=0)))

def get_threshold(raw_data, filter=None, 
                  nexcerpts=None,
                  excerpt_size=None,
                  threshold_std_factor=None,
                  use_single_threshold=True,
                  ):
    
    # We compute the standard deviation of the signal across the excerpts.
    # WARNING: this may use a lot of RAM.
    excerpts = np.vstack(excerpt.data 
        for excerpt in raw_data.excerpts(nexcerpts=nexcerpts, 
                                         excerpt_size=excerpt_size))
    std = np.median(np.abs(excerpts), axis=0) / .6745
    threshold = threshold_std_factor * std
    return np.median(threshold) if use_single_threshold else threshold
    
    