"""Contain default parameters."""

# Filtering
# ---------
filter_low = 500. # Low pass frequency (Hz)
filter_high = 0.95 * .5 * sample_rate
filter_butter_order = 3  # Order of Butterworth filter.

# Chunks
# ------
chunk_size = int(sample_rate)
chunk_overlap = int(.01 * sample_rate)

# Excerpts
# --------
# Uniformly scatter chunks, for computing the threshold from the std of the
# signal across the whole recording.
nexcerpts = 50
excerpt_size = int(1. * sample_rate)
