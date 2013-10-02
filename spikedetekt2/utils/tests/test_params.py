"""Handle user-specified and default parameters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from spikedetekt2 import (python_to_params, params_to_python, 
    load_default_params)

def test_python_to_params():
    python = """
    MYVAR1 = 'myvalue1'
    MYVAR2 = .123
    MYVAR3 = ['myvalue3', .456]
    """.replace('    ', '').strip()
    
    params = python_to_params(python)
    assert params['MYVAR1'] == 'myvalue1'
    assert params['MYVAR2'] == .123
    assert params['MYVAR3'] == ['myvalue3', .456]
    
def test_params_to_python():
    params = dict(
        MYVAR1 = 'myvalue1',
        MYVAR2 = .123,
        MYVAR3 = ['myvalue3', .456])
    
    python = params_to_python(params)
    assert python == """
    MYVAR1 = 'myvalue1'
    MYVAR2 = 0.123
    MYVAR3 = ['myvalue3', 0.456]
    """.replace('    ', '').strip()
    
def test_default_params():
    params_default = load_default_params()
    assert params_default['MYVAR1'] == 'value1'
    

