"""Handle user-specified and default parameters."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from spikedetekt2 import (python_to_pydict, pydict_to_python, 
    load_default_params, get_params)

def test_python_to_params():
    python = """
    MYVAR1 = 'myvalue1'
    MYVAR2 = .123
    MYVAR3 = ['myvalue3', .456]
    """.replace('    ', '').strip()
    
    params = python_to_pydict(python)
    assert params['myvar1'] == 'myvalue1'
    assert params['myvar2'] == .123
    assert params['myvar3'] == ['myvalue3', .456]
    
def test_get_params():
    assert get_params().get('myvar1', None) == 'value1'
    
def test_params_to_python():
    params = dict(
        MYVAR1 = 'myvalue1',
        MYVAR2 = .123,
        MYVAR3 = ['myvalue3', .456])
    
    python = pydict_to_python(params)
    assert python == """
    MYVAR1 = 'myvalue1'
    MYVAR2 = 0.123
    MYVAR3 = ['myvalue3', 0.456]
    """.replace('    ', '').strip()
    
def test_default_params():
    params_default = load_default_params()
    assert params_default['myvar1'] == 'value1'
    

