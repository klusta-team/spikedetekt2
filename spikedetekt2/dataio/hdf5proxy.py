"""Manage experiments."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import json

import numpy as np
import pandas as pd

from selection import select
from spikedetekt2.utils.six import (iteritems, string_types, iterkeys, 
    itervalues)


# -----------------------------------------------------------------------------
# HDF5 proxies
# -----------------------------------------------------------------------------
class ChildProxy(object):
    def __init__(self, parent, name,):
        self._parent = parent
        self._name = name
        
    def __getitem__(self, item):
        return self._parent.__child_getitem__(self._name, item)
        
    def __getattr__(self, item):
        return self._parent.__child_getattr__(self._name, item)
    
    def __repr__(self):
        return "<Child proxy to '{0:s}.{1:s}'>".format(self._parent, self._name)
    
class SelectionProxy(object):
    def __init__(self, parent, selection=None, item=None):
        self._parent = parent
        self._fields = parent._fields
        self._selection = selection
        self._item = item
        
    def __getattr__(self, name):
        table = self._fields[name]
        table_selected = self._selection[table]
        return select((table_selected, name), self._item, doselect=False)
    
    def __repr__(self):
        return "<Selection proxy to '{0:s}[{1:s}]'>".format(
            self._parent, self._item)
    
class HDF5Proxy(object):
    def __init__(self, **fields):
        """Create a proxy object to easily access HDF5 tables.
        
        Arguments:
          * fields: a dictionary {field_name: <pytables.Table object>}
        
        """
        self._fields = fields
        # self.%field% is an object that can be __getitem__ and __getattr__
        # with the callback methods being defined in this parent class.
        for field in iterkeys(self._fields):
            setattr(self, field, ChildProxy(self, field))
        
    def __child_getitem__(self, name, item):
        """Called when self.%name%[item] is called."""
        # Find the table containing the requested field.
        table = self._fields[name]
        return select((table, name), item)
        
    def __child_getattr__(self, name, item):
        """Called when self.%name%.%item% is called."""
        pass

    # def __selection_getattr__(self, name, item):
        # return self.__child_getitem__(name, item)
        
    def __getitem__(self, item):
        # Make the selection for each table.
        selection = {table: table[item] 
            for table in set(itervalues(self._fields))}
        return SelectionProxy(self, selection=selection, item=item)
        
    def __repr__(self):
        return "<Proxy object>"
        
        