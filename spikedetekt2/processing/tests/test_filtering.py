"""Filtering tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from spikedetekt2.processing import apply_filtering, bandpass_filter


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_apply_filtering():
    rate = 10000.
    low, high = 100., 200.
    # Create a signal with small and high frequencies.
    t = np.linspace(0., 1., rate)
    x = np.sin(2*np.pi*low/2*t) + np.cos(2*np.pi*high*2*t)
    # Filter the signal.
    filter = bandpass_filter(low=low, high=high, order=4, rate=rate)
    x_filtered = apply_filtering(x, filter=filter)
    # Check that the bandpass-filtered signal is weak.
    assert np.abs(x[int(2./low*rate):-int(2./low*rate)]).max() >= .9
    assert np.abs(x_filtered[int(2./low*rate):-int(2./low*rate)]).max() <= .1
    
    