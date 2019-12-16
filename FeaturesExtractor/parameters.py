
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

# plag to select which image to plot
FLAG_PLOT_IMG                 = True
FLAG_PLOT_LAMBDA_PLUS         = True
FLAG_PLOT_LAMBDA_MINUS        = True
FLAG_PLOT_LAMBDA_THETA        = False
FLAG_PLOT_IMG_CLIP            = False
FLAG_PLOT_LAMBDA_PLUS_CLIP    = False
FLAG_PLOT_LAMBDA_MINUS_CLIP   = False
FLAG_PLOT_THETA_CLIP          = False
FLAG_PLOT_LAMBDA_PLUS_EDGES   = True
FLAG_PLOT_LAMBDA_MINUS_EDGES  = True



#-----------------------------------------------------------------------
# Quantile to clip the image
#-----------------------------------------------------------------------
CLIP_MIN = 0.001
CLIP_MAX = 0.999

#------------------------------------------------------------------------
# Sigma for Canny edge detection
#------------------------------------------------------------------------
SIGMA_EDGE = 5
FLAG_PLOT_CANNYEDGES    = True

#------------------------------------------------------------------------
# Probability Hough Line Detection
#------------------------------------------------------------------------
LINE_THRESHOLD          = 10
LINE_LENGTH             = 20
LINE_GAP                = 5

#------------------------------------------------------------------------
# Erase line margin
#------------------------------------------------------------------------
LINE_ERASE_MARGIN       = 10


#------------------------------------------------------------------------
# Hough Circle Detection
#------------------------------------------------------------------------
# Detect two radii
HOUGH_RADIUS_MIN        = 15
HOUGH_RADIUS_MAX        = 100
HOUGH_RADIUS_STEP       =  2

# number of circles
NB_HOUGH_CIRCLE_PEAKS   = 3



#------------------------------------------------------------------------
# Circle Validation
#--------------------------------------------------------------------------
# erase circle having signal sum less than this threshold
CIRCLE_SIGNAL_THRESHOLD  = 0.01     # less than 1.0
CIRCLE_MIN_DISTANCE      =  5       # minimum distance between tow circle center



#--------------------------------------------------------------------------
# Profile : get_circles_inprofiles
#-------------------------------------------------------------------------
NBRADIUS                = 1
RADIUSFRACTION          = 0.25

#-----------------------------------------------------------------------
# Test saturation
#-------------------------------------------------------------------------
SATURATION_FRACTION_LEVEL       = 0.95          # quantile of the maximum value
SATURATION_FRACTION_THRESHOLD   = 0.20          # fraction of the number of pixel above SATURATION_FRACTION_LEVEL


#-----------------------------------------------------------------------
# Optimisation to find the global minimum
#-----------------------------------------------------------------------
VIGNETTE_SIZE           = 16   # must be even


#-------------------------------------------------------------------------
# Save output table
#-------------------------------------------------------------------------
FILENAME_LP_SUMMARYTABLE = "lambda_plus_summary_table.txt"

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
