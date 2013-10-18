"""Unit tests for selection module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import Counter

import numpy as np
import pandas as pd

from spikedetekt2.dataio.selection import select, slice_to_indices


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def generate_clusters(indices, nspikes=100):
    """Generate all spikes in cluster 0, except some in cluster 1."""
    # 2 different clusters, with 3 spikes in cluster 1
    clusters = np.zeros(nspikes, dtype=np.int32)
    clusters[indices] = 1
    return clusters
    
def generate_data2D(nspikes=100, ncols=5):
    data = np.random.randn(nspikes, ncols)
    return data

def assert_indexing(indices, n=10):
    values = range(n)
    stop = indices.stop or n
    assert np.array_equal(slice_to_indices(indices, stop=stop), 
                          values[indices])
    

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_slice_to_indices():
    assert_indexing(slice(0, 10, 1))
    
    assert_indexing(slice(None, None, None))
    
    assert_indexing(slice(0, None, None))
    assert_indexing(slice(None, 10, None))
    assert_indexing(slice(None, None, 1))
    
    assert_indexing(slice(None, 10, 1))
    assert_indexing(slice(0, None, 1))
    assert_indexing(slice(0, 10, None))
    
    assert_indexing(slice(1, None, None))
    assert_indexing(slice(1, None, 2))
    
    assert_indexing(slice(1, 9, None))
    assert_indexing(slice(1, 9, 2))
    
    assert_indexing(slice(1, 8, None))
    assert_indexing(slice(1, 8, 2))
    
    assert_indexing(slice(1, 8, None))
    assert_indexing(slice(1, 8, 2))
    assert_indexing(slice(1, 8, 3))
    assert_indexing(slice(1, 8, 4))
    assert_indexing(slice(1, 8, 5))
    assert_indexing(slice(1, 8, 6))
    

def test_select_numpy():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])

def test_select_pandas():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    
    # test selection of Series (1D)
    clusters = pd.Series(clusters)
    assert np.array_equal(select(clusters, [9, 11]), [0, 0])
    assert np.array_equal(select(clusters, [10, 99]), [1, 0])
    assert np.array_equal(select(clusters, [20, 25, 25]), [1, 1, 1])
    
    # test selection of Series (3D)
    clusters = pd.DataFrame(clusters)
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    assert np.array_equal(np.array(select(clusters, [20, 25, 25])).ravel(), [1, 1, 1])
    
    # test selection of Panel (4D)
    clusters = pd.Panel(np.expand_dims(clusters, 3))
    assert np.array_equal(np.array(select(clusters, [9, 11])).ravel(), [0, 0])
    assert np.array_equal(np.array(select(clusters, [10, 99])).ravel(), [1, 0])
    
def test_select_single():
    indices = [10, 20, 25]
    clusters = generate_clusters(indices)
    assert select(clusters, 10) == 1
