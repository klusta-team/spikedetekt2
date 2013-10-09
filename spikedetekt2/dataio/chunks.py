"""Chunking routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# Chunking functions
# -----------------------------------------------------------------------------
#m chops n_samples into chunks according to chunk_size,overlap
#m Overlap probably controls for the artifacts of filtering on the ends of the signal?
def chunk_bounds(n_samples, chunk_size, overlap=0):
    '''
    Returns chunks of the form:
    [ overlap/2 | chunk_size-overlap | overlap/2 ]
    s_start   keep_start           keep_end     s_end
    Except for the first and last chunks which do not have a left/right overlap
    '''
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end-overlap//2
    yield s_start,s_end,keep_start,keep_end
    
    while s_end-overlap+chunk_size < n_samples:
        s_start = s_end-overlap
        s_end = s_start+chunk_size
        keep_start = keep_end
        keep_end = s_end-overlap//2
        yield s_start,s_end,keep_start,keep_end
        
    s_start = s_end-overlap
    s_end = n_samples
    keep_start = keep_end
    keep_end = s_end
    yield s_start,s_end,keep_start,keep_end


