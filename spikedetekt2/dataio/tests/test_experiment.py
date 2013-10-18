"""Experiment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
import tempfile

from spikedetekt2.dataio.experiment import create_experiment
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
def test_experiment_class():
    dir = tempfile.mkdtemp()
    name = 'my experiment'
    mock_experiment(dir=dir, name=name)
    
    # Access the experiment.
    
    
    shutil.rmtree(dir)

