"""Create mock data"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def randint(high):
    return np.random.randint(low=0, high=high)

    
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def random_channel(index=None):
    return {
        'name': 'channel{0:d}'.format(index or randint(10)),
        'ignored': False,
        'position': np.random.randn(2),
        'voltage_gain': 0.,
        'display_threshold': 0.,
        'application_data': {
            'klustaviewa': {},
            'spikedetekt': {},
        },
        'user_data': {},
    }

def random_cluster():
    return {
        'application_data': {
            'klustaviewa': {
                               'cluster_group': randint(4),
                               'color': randint(30),
                           },
            'spikedetekt': {},
        }
    }
    
def random_cluster_group(index=None):
    return {
        'name': 'cluster_group{0:d}'.format(index or randint(10)),
        'application_data': {
            'klustaviewa': {
                               'color': randint(30),
                           },
            'spikedetekt': {},
        },
        'user_data': {},
    }
    
def random_event_type(index=None):
    return {
        'name': 'event_type{0:d}'.format(index or randint(10)),
        'application_data': {
            'klustaviewa': {
                               'color': randint(30),
                           },
            'spikedetekt': {},
        },
        'user_data': {},
    }

