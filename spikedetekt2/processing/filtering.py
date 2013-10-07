"""Filtering routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from spikedetekt2.utils.six.moves import range


# -----------------------------------------------------------------------------
# Signal processing functions
# -----------------------------------------------------------------------------

def bandpass_filter(order=None, rate=None, low=None, high=None):
    """Bandpass filtering."""
    return signal.butter(order,
                         (low/(rate/2.), high/(rate/2.)),
                         'pass')

def apply_filtering(x, filter=None):
    b, a = filter
    # out_arr = np.zeros_like(x)
    # for i_ch in range(x.shape[1]):
        # out_arr[:, i_ch] = signal.filtfilt(b, a, x[:, i_ch]) 
    # return out_arr
    return signal.filtfilt(b, a, x, axis=0)


