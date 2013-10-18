"""Experiment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import tables as tb

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
    name = 'myexperiment'
    mock_experiment(dir=dir, name=name)
    
    # Modify the experiment files.
    path = os.path.join(dir, name + '.kwx')
    f = tb.openFile(path, 'r+')
 
    # TODO
    
    f.close()
    shutil.rmtree(dir)

