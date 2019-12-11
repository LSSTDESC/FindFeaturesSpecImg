
import numpy as np
import os
import astropy.units as units
from astropy import constants as const
import matplotlib as mpl

# These parameters are the default values adapted to Pic du Midi
# To modify them, please create a new config file and load it.

# Paths
mypath = os.path.dirname(__file__)

#-----------------------------------------------------------------------
# Data structure containing the image and its processed images
# - choose the number of usefull processed images
#-----------------------------------------------------------------------
from enum import IntEnum
class IndexImg(IntEnum):
    img                = 0               # the original image
    lambda_plus        = 1               # the image of eigen value of hessian lambda_plus
    lambda_minus       = 2               # the image of eigen value of hessian lambda_minus
    theta              = 3               # the hessian theta value
    img_clip           = 4               # the original image clipped
    lambda_plus_clip   = 5               # the image of eigen value of hessian lambda_plus clipped
    lambda_minus_clip  = 6               # the image of eigen value of hessian lambda_minus
    theta_clip         = 7               # the hessian theta value clipped
    lambda_plus_edges  = 8               # edges by canny of the lambda_plus_clipped  image
    lambda_minus_edges = 9               # edges by canny of the lambda_minus_clipped  image

# number of images
NBIMG                  = 10


#-----------------------------------------------------------------------
# Quantile to clip the image
#-----------------------------------------------------------------------
CLIP_MIN = 0.001
CLIP_MAX = 0.999

#------------------------------------------------------------------------
# Sigma for edge detection
#------------------------------------------------------------------------
SIGMA_EDGE = 5


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
