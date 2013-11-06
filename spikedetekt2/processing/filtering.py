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

def apply_filter(x, filter=None):
    b, a = filter
    # out_arr = np.zeros_like(x)
    # for i_ch in range(x.shape[1]):
        # out_arr[:, i_ch] = signal.filtfilt(b, a, x[:, i_ch]) 
    # return out_arr
    return signal.filtfilt(b, a, x, axis=0)


# -----------------------------------------------------------------------------
# Whitening
# -----------------------------------------------------------------------------
"""
  * Get the first chunk of data
  * Detect spikes the usual way
  * Compute mean on each channel on non-spike data
  * For every pair of channels:
      * estimate the covariance on non-spike data
  * Get the covariance matrix
  * Get its square root C' (sqrtm)
  * Get u*C' + (1-u)*s*Id, where u is a parameter, s the std of non-spike data 
    across all channels
  * Option to save or not whitened data in FIL
  * All spike detection is done on whitened data
  
"""
def get_whitening_matrix(x):
    C = np.cov(x, rowvar=0)
    # TODO

def whiten(x, matrix=None):
    if matrix is None:
        matrix = get_whitening_matrix(x)
    # TODO
    
    


