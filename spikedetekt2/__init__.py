# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import logging

from kwiklib.utils import logger as log
from kwiklib.utils import *
from processing import *
from core import *
from kwiklib.dataio import *


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
LOGGER = log.ConsoleLogger(name='{0:s}.console'.format(APPNAME),
                           print_caller=False)
log.register(LOGGER)

sys.excepthook = log.handle_exception

# Set the logging level.
log.set_level(logging.INFO)

