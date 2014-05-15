"""Launching script."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import os
import sys
import os.path as op

import numpy as np
import tables as tb

from kwiklib import (Experiment, get_params, load_probe, create_files, 
    read_raw, Probe)
from spikedetekt2.core import run


# -----------------------------------------------------------------------------
# Launching script
# -----------------------------------------------------------------------------
def main(prm_filename):
    
    dir, filename = op.split(prm_filename)
    basename, ext = op.splitext(filename)
    assert ext == '.prm'
    assert op.exists(prm_filename)
    
    # Load PRM file.
    prm = get_params(prm_filename)
    nchannels = prm.get('nchannels')
    assert nchannels > 0
    
    # Find PRB path in PRM file, and load it.
    prb_filename = prm.get('prb_file')
    if not op.exists(prb_filename):
        prb_filename = op.join(dir, prb_filename)
    prb = load_probe(prb_filename)
        
    # Find raw data source.
    data_path = prm.get('raw_data_files')
    if not op.exists(data_path):
        data_path = op.join(dir, data_path)
    
    # Create files.
    create_files(basename, dir=dir, prm=prm, prb=prb, create_default_info=True)
    
    # Run SpikeDetekt.
    with Experiment(basename, dir=dir, mode='a') as exp:
        run(read_raw(data_path, nchannels=nchannels), 
            experiment=exp, prm=prm, probe=Probe(prb),
            save_raw=True)

if __name__ == '__main__':
    main(sys.argv[1])
    