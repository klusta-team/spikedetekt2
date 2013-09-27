"""Handle user-specified and default parameters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from spikedetekt2 import python_to_params, params_to_python

def test_python_to_params():
    python = """
    MYVAR1 = 'myvalue1'
    MYVAR2 = 0.123
    MYVAR3 = ['myvalue3', .456]
    """.replace('    ', '')
    
    params = python_to_params(python)
    assert params['MYVAR1'] == 'myvalue1'
    assert params['MYVAR2'] == 0.123
    assert params['MYVAR3'] == ['myvalue3', .456]
    
    
    

