"""Create mock data"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.dataio.kwik_creation import *


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
    