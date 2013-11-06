"""Handle user-specified and default parameters."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.utils import python_to_pydict, to_lower, get_pydict
from six import string_types, iteritems


# -----------------------------------------------------------------------------
# Python script <==> dictionaries conversion
# -----------------------------------------------------------------------------
def get_params(filename=None, **params):
    return get_pydict(filename=filename, 
                      pydict_default=load_default_params(),
                      **params)


# -----------------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------------
def load_default_params():
    folder = os.path.dirname(os.path.realpath(__file__))
    params_default_path = os.path.join(folder, 'params_default.py')
    with open(params_default_path, 'r') as f:
        params_default_python = f.read()
    params_default = python_to_pydict(params_default_python)
    return to_lower(params_default)

