"""Launching script."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import os
import sys
import os.path as op
import tempfile
import argparse

import numpy as np
import tables as tb

from kwiklib import (Experiment, get_params, load_probe, create_files, 
    read_raw, Probe, convert_dtype, read_clusters,
    files_exist, add_clustering, delete_files)
from spikedetekt2.core import run


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _load_files_info(prm_filename, dir=None):
    dir_, filename = op.split(prm_filename)
    dir = dir or dir_
    basename, ext = op.splitext(filename)
    if ext == '':
        ext = '.prm'
    prm_filename = op.join(dir, basename + ext)
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
    data = prm.get('raw_data_files')
    if isinstance(data, basestring):
        if data.endswith('.dat'):
            data = [data]
    if isinstance(data, list):
        for i in range(len(data)):
            if not op.exists(data[i]):
                data[i] = op.join(dir, data[i])
        
    experiment_name = prm.get('experiment_name')
    
    return dict(prm=prm, prb=prb, experiment_name=experiment_name, nchannels=nchannels,
                data=data, dir=dir)
    

# -----------------------------------------------------------------------------
# SpikeDetekt
# -----------------------------------------------------------------------------
def run_spikedetekt(prm_filename, dir=None, debug=False):
    info = _load_files_info(prm_filename, dir=dir)
    experiment_name = info['experiment_name']
    prm = info['prm']
    prb = info['prb']
    data = info['data']
    dir = dir or info['dir']
    nchannels = info['nchannels']
    
    # Make sure spikedetekt does not run if the .kwik file already exists
    # (i.e. prevent running it twice on the same data)
    assert not files_exist(experiment_name, dir=dir, types=['kwik']), "The .kwik file already exists, please use the --overwrite option."
    
    # Create files.
    create_files(experiment_name, dir=dir, prm=prm, prb=prb, 
                 create_default_info=True, overwrite=False)
    
    # Run SpikeDetekt.
    with Experiment(experiment_name, dir=dir, mode='a') as exp:
        run(read_raw(data, nchannels=nchannels), 
            experiment=exp, prm=prm, probe=Probe(prb),
            _debug=debug)


# -----------------------------------------------------------------------------
# KlustaKwik
# -----------------------------------------------------------------------------
PARAMS_KK = dict(
    MaskStarts = 100,
    #MinClusters = 100 ,
    #MaxClusters = 110,
    MaxPossibleClusters =  500,
    FullStepEvery = 10,
    MaxIter = 10000,
    RandomSeed =  654,
    Debug = 0,
    SplitFirst = 20 ,
    SplitEvery = 100 ,
    PenaltyK = 0,
    PenaltyKLogN = 1,
    Subset = 1,
    PriorPoint = 1,
    SaveSorted = 0,
    SaveCovarianceMeans = 0,
    UseMaskedInitialConditions = 1,
    AssignToFirstClosestMask = 1,
    UseDistributional = 1,
)

def write_mask(mask, filename, fmt="%f"):
    with open(filename, 'w') as fd:
        fd.write(str(mask.shape[1])+'\n') # number of features
        np.savetxt(fd, mask, fmt=fmt)

def write_fet(fet, filepath):
    with open(filepath, 'w') as fd:
        #header line: number of features
        fd.write('%i\n' % fet.shape[1])
        #next lines: one feature vector per line
        np.savetxt(fd, fet, fmt="%i")

def save_old(exp, shank, dir=None):
    chg = exp.channel_groups[shank]
            
    # Create files in the old format (FET and FMASK)
    fet = chg.spikes.features_masks[...]
    if fet.ndim == 3:
        masks = fet[:,:,1]  # (nsamples, nfet)
        fet = fet[:,:,0]  # (nsamples, nfet)
    else:
        masks = None
    res = chg.spikes.time_samples[:]
    
    times = np.expand_dims(res, axis =1)
    masktimezeros = np.zeros_like(times)
    
    fet = convert_dtype(fet, np.int16)
    fet = np.concatenate((fet, times),axis = 1)
    mainfetfile = os.path.join(dir, exp.name + '.fet.' + str(shank))
    write_fet(fet, mainfetfile)
    
    if masks is not None:
        fmasks = np.concatenate((masks, masktimezeros),axis = 1)
        fmaskfile = os.path.join(dir, exp.name + '.fmask.' + str(shank))
        write_mask(fmasks, fmaskfile, fmt='%f')
    
def run_klustakwik(filename, dir=None, **kwargs):
    # Open the KWIK files in append mode so that we can write the clusters.
    with Experiment(filename, dir=dir, mode='a') as exp:
        name = exp.name
        shanks = exp.channel_groups.keys()
        
        # Set the KlustaKwik parameters.
        params = PARAMS_KK.copy()
        for key in PARAMS_KK.keys():
            # Update the PARAMS_KK keys if they are specified directly
            # but ignore the kwargs keys that do not appear in PARAMS_KK.
            params[key] = kwargs.get(key.lower(), params[key])
            
        # Switch to temporary directory.
        start_dir = os.getcwd()
        tmpdir = os.path.join(start_dir, '_klustakwik')
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
        os.chdir(tmpdir)
        
        for shank in shanks:
            # chg = exp.channel_groups[shank]
            
            save_old(exp, shank, dir=tmpdir)
            
            # Generate the command for running klustakwik.
            cmd = 'klustakwik {name} {shank} {params}'.format(
                name=name,
                shank=shank,
                params=' '.join(['-{key} {val}'.format(key=key, val=str(val))
                                    for key, val in params.iteritems()]),
            )
            
            # Run KlustaKwik.
            os.system(cmd)
            
            # Read back the clusters.
            clu = read_clusters(name + '.clu.' + str(shank))
            
            # Put the clusters in the kwik file.
            add_clustering(exp._files, channel_group_id=str(shank), name='original',
                           spike_clusters=clu, overwrite=True)
            add_clustering(exp._files, channel_group_id=str(shank), name='main',
                           spike_clusters=clu, overwrite=True)
        
        # Switch back to original dir.
        os.chdir(start_dir)
        

# -----------------------------------------------------------------------------
# All-in-one script
# -----------------------------------------------------------------------------
def run_all(prm_filename, dir=None, debug=False, overwrite=False):
    info = _load_files_info(prm_filename, dir=dir)
    experiment_name = info['experiment_name']
    prm = info['prm']
    prb = info['prb']
    data = info['data']
    nchannels = info['nchannels']
    
    if files_exist(experiment_name, dir=dir):
        if overwrite:
            delete_files(experiment_name, dir=dir, types=('kwik', 'kwx', 'high.kwd', 'low.kwd'))
        else:
            print(("The files already exist, delete them first "
                  "if you want to run the process again, or user the "
                  "--overwrite option."))
    
    run_spikedetekt(prm_filename, dir=dir, debug=debug)
    run_klustakwik(experiment_name, dir=dir, **prm)
        
def main():
    parser = argparse.ArgumentParser(description='Run spikedetekt and klustakwik.')
    parser.add_argument('prm_file',
                       help='.prm filename')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='run the first few seconds of the data for debug purposes')
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help='overwrite the KWIK files is they already exist')

    args = parser.parse_args()
    
    run_all(args.prm_file, debug=args.debug, overwrite=args.overwrite)
        
if __name__ == '__main__':
    main()