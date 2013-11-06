"""Handle user-specified and default parameters."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from six import string_types, iteritems


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _to_str(val):
    """Get a string representation of any Python variable."""
    if isinstance(val, string_types):
        return "'{0:s}'".format(val)
    else:
        return str(val)

def _to_lower(d):
    return {key.lower(): val for key, val in iteritems(d)}
    

# -----------------------------------------------------------------------------
# Python script <==> dictionaries conversion
# -----------------------------------------------------------------------------
def python_to_params(script_contents):
    """Load a Python script with parameters into a dictionary."""
    params = {}
    exec script_contents in {}, params
    return _to_lower(params)
    
def params_to_python(params):
    """Convert a parameters dictionary into a Python script."""
    return "\n".join(["{0:s} = {1:s}".format(key, _to_str(val))
        for key, val in sorted(params.iteritems())])

def get_params(filename=None, **params):
    params_default = load_default_params()
    params_final = params_default.copy()
    if isinstance(filename, string_types):
        # Path to PRM file.
        with open(filename, 'r') as f:
            params_prm = python_to_params(f.read())
            params_final.update(params_prm)
    params_final.update(params)
    return _to_lower(params_final)


# -----------------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------------
def load_default_params():
    folder = os.path.dirname(os.path.realpath(__file__))
    params_default_path = os.path.join(folder, 'params_default.py')
    with open(params_default_path, 'r') as f:
        params_default_python = f.read()
    params_default = python_to_params(params_default_python)
    return _to_lower(params_default)

