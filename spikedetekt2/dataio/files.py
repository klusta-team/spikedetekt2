"""Manage file names."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os


# -----------------------------------------------------------------------------
# File names
# -----------------------------------------------------------------------------
def generate_filenames(name):
    """Generate a list of filenames for the different files in a given 
    experiment, which name is given."""
    name = os.path.splitext(name)[0]
    return {
        'kwik': name + '.kwik',
        'kwx': name + '.kwx',
        'kwe': name + '.kwe',
        'kwd': {
                 'raw': name + '.raw.kwd',
                 'high': name + '.high.kwd',
                 'low': name + '.low.kwd',
                }
    }
    
    