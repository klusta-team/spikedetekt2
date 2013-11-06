"""Graph tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.processing.graph import (connected_components, 
    get_component, _to_tuples, _to_list)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def _clip(x, m, M):
    return [_ for _ in x if m <= _ < M]

n = 5
graph = {i: set(_clip([i-1, i+1], 0, n))
            for i in range(n)}

def test_get_component():
    chunk = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1]
    ])
    position = (2, 3)
    comp = _to_list(get_component(chunk, position))
    assert (1, 1) not in comp
    assert (1, 2) in comp
    assert (2, 0) not in comp
    assert (2, 2) in comp
    assert (2, 3) in comp
    assert (3, 3) in comp
    
def test_connected_components_1():
    chunk = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1]
    ])
    components = connected_components(chunk, graph=graph)
    for c in components:
        print c
    # assert len(components) == 7
    
def test_connected_components_2():
    chunk = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1]
    ])
    components = connected_components(chunk, graph=graph, join_size=1)
    for c in components:
        print c
    # assert len(components) == 7
    