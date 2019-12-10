
import numpy as np
import os
import astropy.units as units
from astropy import constants as const
import matplotlib as mpl

# These parameters are the default values adapted to CTIO
# To modify them, please create a new config file and load it.

# Paths
mypath = os.path.dirname(__file__)



# Plotting
PAPER = False
LINEWIDTH = 2
PLOT_DIR = 'plots'
SAVE = False

# Verbosity
VERBOSE = False
DEBUG = True
MY_FORMAT = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"

# Plots
DISPLAY = True
if os.environ.get('DISPLAY', '') == '':
    mpl.use('agg')
    DISPLAY = False
