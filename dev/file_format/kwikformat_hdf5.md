Overview
--------

File format specification
-------------------------

  * The data are stored in the following files:
      
      * the **KWIK** file is the main file, it contains the structure of all data bits, the **metadata**, aesthetic information (JSON)
      * the **KWX** file contains the **spiking data** (HDF5)
      * the **KWD** files contain the **raw/filtered recordings** (HDF5)
      * the **KWE** file contain the **events** (HDF5)

  * All files contain a **version number** in `/` (VERSION attribute), which is an integer equal to 2 now.

  * The input files the user provides to the programs to generate these data are:
  
      * the **raw data** coming out from the acquisition system, in any proprietary format (NS5, etc.)
      * processing parameters (PRM file) and description of the probe (PRB file)
  

### KWIK

This JSON text file contains all metadata related to the experiment, and aesthetic information about channel and cluster colors, the cluster groups, the channel positions and scaling, etc.
    
    name
    application_data
        spikedetekt
            concatenation_list
            ...
    user_data
    channel_groups
        []
            name
            graph
            application_data
            user_data
            channels
                []
                    name
                    ignored
                    position
                    voltage_gain
                    display_threshold
                    application_data
                        klustaviewa
                        spikedetekt
                    user_data
            spikes
                hdf5_path
                    main = '{KWX}/channel_groups/channel_groupX/spikes'
                    clusters = '{KWX}/channel_groups/channel_groupX/clusters'
                    waveforms = '{KWX}/channel_groups/channel_groupX/waveforms'
            clusters
                []
                    application_data
                        klustaviewa
                            cluster_group
                            color
            cluster_groups
                []
                    name
                    application_data
                        klustaviewa
                            color
                    user_data
    recordings
        []
            name
            user_data
            data
                hdf5_path
                    raw = '{KWD}/data_raw/recording0'
                    high_pass = '{KWD}/data_high/recording0'
                    low_pass = '{KWD}/data_low/recording0'
            start_time
            start_sample
            sample_rate
            band_low
            band_high
            bit_depth
    events
        hdf5_path = '{KWE}/events'
    event_types
        []
            name
            application_data
                klustaviewa
                    color
            user_data


### KWX

The HDF5 **KWX** file contains all spiking information.
 
  * `/channel_groups/channel_groupX/` ( *X* being the channel_group index, starting from 1): *group* with the spikes detected on that channel_group.

  * `/channel_groups/channel_groupX/clusters`: *table*, one row = one spike, and the following columns:
      * `cluster_auto`: UInt32, the cluster number (max ~ 10^10), obtained after the automatic clustering stage
      * `cluster_manual`: UInt32, the cluster number (max ~ 10^10), obtained after the manual stage
  
  * `/channel_groups/channel_groupX/spikes`: *table*, one row = one spike, and the following columns:
      * `time`: UInt64, spike time, in number of samples (max ~ 10^19)
      * `features`: Float32(nfet,), a vector with the spike features, typically nfet=nchannels*fetdim+nextrafet, with fetdim the number of principal components per channel
      * `masks`: UInt8(nfet,), a vector with the masks, 0=totally masked, 255=unmasked
  
  * `/channel_groups/channel_groupX/waveforms`: *table*, one row = one spike, and the following columns:
      * `waveform_filtered`: Int16(nsamples*nchannels,), a vector with the high-pass filtered spike waveform. Stride order: sample first, channel second.
      * `waveform_raw`: Int16(nsamples*nchannels,), a vector with the raw spike waveform.
  

### KWD

The HDF5 **KWD** files contain all non-spiking (raw or filtered) information.

  * `.raw.KWD`:
      * `/data_raw`: [EArray](http://pytables.github.io/usersguide/libref/homogenous_storage.html#the-earray-class)(Int16, (duration*freq, nchannels)) with the raw data on all channels
  
  * `.high.KWD`:
      * `/data_high`: high-pass filered data
  
  * `.low.KWD`:
      * `/data_low`: low-pass filered data

      
### KWE

The HDF5 **KWE** file contains the events.

  * `/events`: *table*, one row = one event, and the following columns:
      * `sample`: UInt64, the time sample of the event
      * `event_type`: the ID of the event type, corresponding to the `event_types` table
      * `recordingID`: the recording ID
      

### PRB

This JSON text file describes the probe used for the experiment: its geometry, its topology, and the dead channels.

    {
        "channel_groups": 
            [
                {
                    "channel_group_index": 1,
                    "channels": [0, 1, 2, 3],
                    "graph": [[0, 1], [2, 3], ...],
                    "geometry": {"0": [0.1, 0.2], "1": [0.3, 0.4], ...}
                },
                {
                    "channel_group_index": 2,
                    "channels": [4, 5, 6, 7],
                    "graph": [[4, 5], [6, 7], ...],
                    "geometry": {"4": [0.1, 0.2], "5": [0.3, 0.4], ...}
                }
            ]
    }


### PRM

This text file (written in a tiny subset of Python) contains all parameters necessary for the programs to process, open and display the data. Each line is either a comment (starting with #) or a `VARNAME = VALUE` where VARNAME is the variable name, and VALUE is either a number, a string (within quotes), or a list of those. This structure ensures that it's easy to read/write this file programmatically.

This file is converted into JSON before being saved within the KWIK file.

    EXPERIMENT_NAME = 'myexperiment'
    RAW_DATA_FILES = ['n6mab041109blahblah1.ns5', 'n6mab041109blahblah2.ns5']
    PRB_FILE = 'buzsaki32.probe'
    NCHANNELS = 32
    SAMPLING_FREQUENCY = 20000.
    IGNORED_CHANNELS = [2, 5]
    NBITS = 16
    VOLTAGE_GAIN = 10.
    WAVEFORMS_NSAMPLES = 20  # or a dictionary {channel_group: nsamples}
    FETDIM = 3  # or a dictionary {channel_group: fetdim}
    # ...
    
    # SpikeDetekt parameters file
    # ...

