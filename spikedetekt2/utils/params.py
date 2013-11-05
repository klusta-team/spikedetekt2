"""Handle user-specified and default parameters."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import six


# -----------------------------------------------------------------------------
# Python script <==> dictionaries conversion
# -----------------------------------------------------------------------------
def python_to_params(script_contents):
    """Load a Python script with parameters into a dictionary."""
    params = {}
    exec script_contents in {}, params
    return params
    
def _to_str(val):
    """Get a string representation of any Python variable."""
    if isinstance(val, six.string_types):
        return "'{0:s}'".format(val)
    else:
        return str(val)
    
def params_to_python(params):
    """Convert a parameters dictionary into a Python script."""
    return "\n".join(["{0:s} = {1:s}".format(key, _to_str(val))
        for key, val in sorted(params.iteritems())])

def get_params(filename=None, **params):
    params_default = load_default_params()
    if isinstance(filename, six.string_types):
        # Path to PRM file.
        with open(filename, 'r') as f:
            params_prm = python_to_params(f.read())
            params_default.update(params_prm)
    params_default.update(params)
    return params_default

# -----------------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------------
def load_default_params():
    folder = os.path.dirname(os.path.realpath(__file__))
    params_default_path = os.path.join(folder, 'params_default.py')
    with open(params_default_path, 'r') as f:
        params_default_python = f.read()
    params_default = python_to_params(params_default_python)
    return params_default

