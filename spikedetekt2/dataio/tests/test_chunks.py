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
    chunks = chunk_bounds(200, 100, overlap=20)
    
    assert next(chunks) == (0, 100, 0, 90)
    assert next(chunks) == (80, 180, 90, 170)
    assert next(chunks) == (160, 200, 170, 200)
    
