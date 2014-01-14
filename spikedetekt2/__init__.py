# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import logging

import spikedetekt2.utils.logger as log
from utils import *
from processing import *
from core import *
from dataio import *


# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------
__version__ = '0.3.0dev'

APPNAME = 'spikedetekt'

ABOUT = """Spikedetekt detekt spikes.

This software was developed by Cyrille Rossant, Shabnam Kadir, Dan Goodman, Kenneth Harris in the Cortical Processing Laboratory at UCL (http://www.ucl.ac.uk/cortexlab)."""


# -----------------------------------------------------------------------------
# Loggers
# -----------------------------------------------------------------------------
LOGGERS = {}
log.LOGGERS = LOGGERS
# Console logger.
LOGGER = log.ConsoleLogger(name='{0:s}.console'.format(APPNAME))
log.register(LOGGER)

sys.excepthook = log.handle_exception

# Set the logging level.
log.set_level(logging.DEBUG)

