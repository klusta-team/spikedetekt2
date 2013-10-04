"""Unit tests for the wrap module."""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from spikedetekt2.utils import wrap


# -----------------------------------------------------------------------------
# Wrap tests
# -----------------------------------------------------------------------------
def test_wrap_1():
    d = dict(key1=1, key2=2)
    dw = wrap(d)
    
    assert dw.key1 == 1
    assert dw.key2 == 2
    
def test_wrap_2():
    d = dict(key1=[0, 1], key2=[0, 10])
    dw = wrap(d)
    
    assert dw.key1 == [0, 1]
    assert dw.key2 == [0, 10]
    
    assert dw.key1[0] == 0
    assert dw.key1[1] == 1
    assert dw.key2[0] == 0
    assert dw.key2[1] == 10

def test_wrap_3():
    d = dict(key1=[0, 1], key2=dict(skey1=['0', '1'], skey2=['0', '10']))
    dw = wrap(d)
    
    assert dw.key1 == [0, 1]
    
    assert dw.key2[0].skey1 == '0'
    assert dw.key2[0].skey2 == '0'
    assert dw.key2[1].skey1 == '1'
    assert dw.key2[1].skey2 == '10'
    
    assert dw.key2.skey1 == ['0', '1']
    assert dw.key2.skey2 == ['0', '10']
    
def test_wrap_4():
    d = {'key1': 
            [
                {
                    'skey1': [0, 1], 
                    'skey2': [0, 10]
                },
                
                {
                    'skey1': [10, 11], 
                    'skey2': [10, 20]
                },   
            ]
        }
    dw = wrap(d)
    
    assert dw.key1[0].skey1 == [0, 1]
    assert dw.key1[0].skey2 == [0, 10]
    
    assert dw.key1[1].skey1 == [10, 11]
    assert dw.key1[1].skey2 == [10, 20]
    
def test_wrap_add_1():
    d = dict(key1=[0, 1], key2=[0, 10], key3={
        'skey1': [10, 11],
        'skey2': [20, 21],
        })
    dw = wrap(d)
    
    dw.key1.append(2)
    assert dw.key1 == [0, 1, 2]
    
    assert dw.key3.skey1 == [10, 11]
    dw.key3.append({'skey1': 12, 'skey2': 22})
    assert dw.key3.skey1 == [10, 11, 12]
    assert dw.key3.skey2 == [20, 21, 22]

def test_wrap_add_2():
    d = dict(key1=[0, 1], key2=np.zeros(2), key3=np.zeros((2, 10)))
    dw = wrap(d)
    
    dw.append({'key1': 2, 
               'key2': 0,
               'key3': np.zeros((1, 10))})
    assert dw.key1 == [0, 1, 2]
    assert np.array_equal(dw.key2, np.zeros(3))
    assert np.array_equal(dw.key3, np.zeros((3, 10)))
    