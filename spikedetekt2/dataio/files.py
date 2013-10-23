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
        
# def get_file(f, type=None):
    # """Return an opened PyTables.File instance, from the filename, basename, or
    # the instance itself. type can be any of 'kwik', 'kwx', 'raw.kwd', 
    # 'low.kwd', 'high.kwd'."""
    # if not type:
        # type = 'kwik'
    # if isinstance(f, tb.File):
        # assert f.filename.endswith('.' + type)
        # return f
    # elif isinstance(f, string_types):
        # bn = get_basename(f)
        # filenames = get_filenames(name)
        # filename = filenames.get(type, None)
        # return tb.openFile(filename, 'r')
        
        
# -----------------------------------------------------------------------------
# Opening functions
# -----------------------------------------------------------------------------
def open_file(path):
    try:
        return tb.openFile(path, 'r')
    except:
        return None

def open_files(name):
    filenames = get_filenames(name)
    return {type: open_file(filenames[type]) for type in FILE_TYPES}


        