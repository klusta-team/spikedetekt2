"""This module provides functions used to update HDF5 files."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import warnings
from collections import OrderedDict, Iterable

import numpy as np
import tables as tb

from spikedetekt2.dataio.utils import save_json
from spikedetekt2.utils.six import iteritems

# Disable PyTables' NaturalNameWarning due to nodes which have names starting 
# with an integer.
warnings.simplefilter('ignore', tb.NaturalNameWarning)


# -----------------------------------------------------------------------------
# File updates
# -----------------------------------------------------------------------------

