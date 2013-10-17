"""Experiment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

from spikedetekt2.dataio.experiment import create_experiment


# -----------------------------------------------------------------------------
# Experiment tests
# -----------------------------------------------------------------------------
def test_create_experiment():
    """Create an experiment (write some files)."""
    name = 'my experiment'
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
    
    exp = create_experiment(name=name, 
        channel_groups_info=channel_groups_info,
        event_types_info=event_types_info,
        recordings_info=recordings_info)
    
    
    
    