"""Unit tests for the wrap module."""
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
    
def test_wrap_add():
    d = dict(key1=[0, 1], key2=[0, 10])
    dw = wrap(d)
    
    dw.key1.append(2)
    assert dw.key1 == [0, 1, 2]
    
    
    
