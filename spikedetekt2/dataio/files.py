"""Manage file names."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import tables as tb
from spikedetekt2.utils.six import string_types


# -----------------------------------------------------------------------------
# File names
# -----------------------------------------------------------------------------
def get_filenames(name):
    """Generate a list of filenames for the different files in a given 
    experiment, which name is given."""
    name = os.path.splitext(name)[0]
    return {
        'kwik': name + '.kwik',
        'kwx': name + '.kwx',
        'raw.kwd':  name + '.raw.kwd',
        'high.kwd':  name + '.high.kwd',
        'low.kwd':  name + '.low.kwd',
    }
    
def get_basename(path):
    bn = os.path.basename(path)
    bn = os.path.splitext(bn)[0]
    if bn.split('.')[-1] in ('raw', 'high', 'low'):
        return os.path.splitext(bn)[0]
    else:
        return bn
        
def get_file(f, type=None):
    """Return an opened PyTables.File instance, from the filename, basename, or
    the instance itself. type can be any of 'kwik', 'kwx', 'raw.kwd', 
    'low.kwd', 'high.kwd'."""
    if not type:
        type = 'kwik'
    if isinstance(f, tb.File):
        assert f.filename.endswith('.' + type)
        return f
    elif isinstance(f, string_types):
        bn = get_basename(f)
        filenames = get_filenames(name)
        return filenames.get(type, None)
        
        
        