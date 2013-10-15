"""Mock data tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from spikedetekt2.dataio.mock import *


# -----------------------------------------------------------------------------
# Chunking Tests
# -----------------------------------------------------------------------------
def test_mock_1():
    assert randint(10) < 10
    
    channel = random_channel(3)
    assert channel['name'] == 'channel3'
    
    cluster = random_cluster()
    cluster_group = random_cluster_group()
    event_type = random_event_type()
    
    