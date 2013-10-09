"""Chunking tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from spikedetekt2.dataio import chunk_bounds


# -----------------------------------------------------------------------------
# Chunking Tests
# -----------------------------------------------------------------------------
def test_chunk_bounds():
    # for i, (s0, s1, keep0, keep1) in enumerate(
            # chunk_bounds(1010, 100, overlap=20)):
        
    chunks = chunk_bounds(1010, 100, overlap=20)
    
    assert next(chunks) == (0, 100, 0, 90)
    assert next(chunks) == (80, 180, 90, 170)
    
