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
    """Compute the threshold from the standard deviation of the filtered signal
    across many uniformly scattered excerpts of data."""
    
    # We compute the standard deviation of the signal across the excerpts.
    # WARNING: this may use a lot of RAM.
    excerpts = np.vstack(
        # Filter each excerpt.
        apply_filter(excerpt.data, filter=filter)
            for excerpt in raw_data.excerpts(nexcerpts=nexcerpts, 
                                             excerpt_size=excerpt_size))
    # Get the median of all samples in all excerpts,
    # on all channels...
    if use_single_threshold:
        median = np.median(np.abs(excerpts))
    # ...or independently for each channel.
    else:
        median = np.median(np.abs(excerpts), axis=0)
    
    # Compute the threshold from the median.
    std = median / .6745
    threshold = threshold_std_factor * std
    
    return threshold
    
    