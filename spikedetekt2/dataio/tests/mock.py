"""Create mock data"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio.kwik_creation import *
from spikedetekt2.dataio.experiment import create_experiment


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def randint(high):
    return np.random.randint(low=0, high=high)


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def mock_kwik():
    return create_kwik_main(
        name='my experiment',
        channel_groups=[
            create_kwik_channel_group(
                ichannel_group=0,
                name='my channel group',
                graph=[[0,1], [1, 2]],
                channels=[
                    create_kwik_channel(
                        name='my first channel',
                        ignored=False,
                        position=[0., 0.],
                        voltage_gain=10.,
                        display_threshold=None),
                    create_kwik_channel(
                        name='my second channel',
                        ignored=True,
                        position=[1., 1.],
                        voltage_gain=20.,
                        display_threshold=None),
                    create_kwik_channel(
                        name='my third channel',
                        ignored=False,
                        position=[0., 2.],
                        voltage_gain=30.,
                        display_threshold=None),
                ],
                cluster_groups=[
                    create_kwik_cluster_group(color=2, name='my cluster group',
                        clusters=[
                             create_kwik_cluster(color=4),
                        ])
                ],
            )
        ],
        recordings=[
            create_kwik_recording(
                irecording=0,
                start_time=0.,
                start_sample=0,
                sample_rate=20000.,
                band_low=100.,
                band_high=3000.,
                bit_depth=16),
        ],
        event_types=[
            create_kwik_event_type(
                color=3,
            ),
        ],
    )
    
def mock_experiment(dir=None, name=None):
    channel_groups_info = [
            {
                'name': 'first channel group',
                'graph': [[0, 1]],
                'channels': [
                    {'name': 'first channel', 'position': (0., 0.)},
                    {'name': 'second channel', 'position': (1., 2.)},
                ],
                'cluster_groups': [
                    {'name': 'first cluster group',
                     'clusters': [
                        {'color': 3}
                     ]
                    }
                ],
            }
        ]
    
    event_types_info = [
        {'name': 'my event type', 'color': 2},
    ]
    
    recordings_info = [
        {
            'name': 'my first recording',
            'start_time': 0.,
            'start_sample': 0,
            'sample_rate': 20000.,
            'band_low': 100.,
            'band_high': 3000.,
            'bit_depth': 16,
        }
    ]
    
    spikedetekt_params = {
        'nwavesamples': 20,
        'fetdim': 3,
    }
    
    create_experiment(
        name=name, 
        dir=dir,
        channel_groups_info=channel_groups_info,
        event_types_info=event_types_info,
        recordings_info=recordings_info,
        spikedetekt_params=spikedetekt_params)
    
    
    
    