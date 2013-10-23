"""Manage file names."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.utils.six import string_types


# -----------------------------------------------------------------------------
# File names
# -----------------------------------------------------------------------------
RAW_TYPES = ('raw.kwd', 'high.kwd', 'low.kwd')
FILE_TYPES = ('kwik', 'kwx') + RAW_TYPES

def get_filenames(name, dir=None):
    """Generate a list of filenames for the different files in a given 
    experiment, which name is given."""
    if dir is None:
        dir = os.path.dirname(os.path.realpath(__file__))
    name = os.path.splitext(name)[0]
    return {type: os.path.join(dir, name + '.' + type) for type in FILE_TYPES}
    
def get_basename(path):
    bn = os.path.basename(path)
    bn = os.path.splitext(bn)[0]
    if bn.split('.')[-1] in ('raw', 'high', 'low'):
        return os.path.splitext(bn)[0]
    else:
        return bn



        