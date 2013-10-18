"""Experiment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
import tempfile

from spikedetekt2.dataio.experiment import create_experiment, Channels
from spikedetekt2.dataio.tests.mock import mock_experiment


# -----------------------------------------------------------------------------
# Experiment creation tests
# -----------------------------------------------------------------------------
def test_create_experiment():
    """Create an experiment (write some files)."""
    dir = tempfile.mkdtemp()
    name = 'my experiment'
    mock_experiment(dir=dir, name=name)
    shutil.rmtree(dir)
    
    
# -----------------------------------------------------------------------------
# Experiment class tests
# -----------------------------------------------------------------------------
def test_experiment_channels():
    channels = [
        {
            'name': 'channel' + str(i),
            'application_data': {
                'klustaviewa': i*10,
                'spikedetekt': i*100,
            }
        } 
        for i in range(100)]
    import pandas as pd
    df = pd.DataFrame.from_dict(channels)
    # df = pd.Panel.from_dict(channels).to_frame()
    print df.ix[::2].application_data
    

def test_experiment_class():
    dir = tempfile.mkdtemp()
    name = 'my experiment'
    mock_experiment(dir=dir, name=name)
    
    # Access the experiment.
    
    
    shutil.rmtree(dir)

