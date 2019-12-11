import os

from skimage.feature import hessian_matrix
import numpy as np
from astropy.stats import sigma_clip
from astropy.io import fits


import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cmx

import astropy
from astropy.visualization.mpl_normalize import (ImageNormalize,MinMaxInterval,PercentileInterval,SqrtStretch,LogStretch)
from astropy.visualization.wcsaxes import SphericalCircle

import time
from datetime import date

#-------------------------------------------------------------------------------------
def ensure_dir(directory_name):
    """Ensure that *directory_name* directory exists. If not, create it.

    Parameters
    ----------
    directory_name: str
        The directory name.

    Examples
    --------
    >>> ensure_dir('tests')
    >>> os.path.exists('tests')
    True
    >>> ensure_dir('tests/mytest')
    >>> os.path.exists('tests/mytest')
    True
    >>> os.rmdir('./tests/mytest')
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
#-------------------------------------------------------------------------------------

def clip_array(in_array,themin,themax):
    amin = np.quantile(in_array, themin)
    amax = np.quantile(in_array, themax)
    out_array = np.clip(in_array, amin, amax)
    return out_array


#-------------------------------------------------------------------------------------
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    For example for the PSF

    x=pixel number
    y=Intensity in pixel

    values-x
    weights=y=f(x)

    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
    return average, np.sqrt(variance)
#-------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------
def hessian_and_theta(data, margin_cut=1):
    # compute hessian matrices on the image
    Hxx, Hxy, Hyy = hessian_matrix(data, sigma=3, order='xy')
    lambda_plus = 0.5 * ((Hxx + Hyy) + np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy * Hxy))
    lambda_minus = 0.5 * ((Hxx + Hyy) - np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy * Hxy))
    theta = 0.5 * np.arctan2(2 * Hxy, Hyy - Hxx) * 180 / np.pi
    # remove the margins
    lambda_minus = lambda_minus[margin_cut:-margin_cut, margin_cut:-margin_cut]
    lambda_plus = lambda_plus[margin_cut:-margin_cut, margin_cut:-margin_cut]
    theta = theta[margin_cut:-margin_cut, margin_cut:-margin_cut]
    return lambda_plus, lambda_minus, theta
#-------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------

def load_fits(file_name, hdu_index=0):
    hdu_list = fits.open(file_name)
    header = hdu_list[0].header
    data = hdu_list[hdu_index].data
    hdu_list.close()  # need to free allocation for file descripto
    return header, data
#----------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------
def plot_image_simple(ax, data, scale="lin", title="", units="Image units", cmap="jet",
                      target_pixcoords=None, vmin=None, vmax=None, aspect=None, cax=None):
    """Simple function to plot a spectrum with error bars and labels.

    Parameters
    ----------
    ax: Axes
        Axes instance to make the plot
    data: array_like
        The image data 2D array.
    scale: str
        Scaling of the image (choose between: lin, log or log10) (default: lin)
    title: str
        Title of the image (default: "")
    units: str
        Units of the image to be written in the color bar label (default: "Image units")
    cmap: colormap
        Color map label (default: None)
    target_pixcoords: array_like, optional
        2D array  giving the (x,y) coordinates of the targets on the image: add a scatter plot (default: None)
    vmin: float
        Minimum value of the image (default: None)
    vmax: float
        Maximum value of the image (default: None)
    aspect: str
        Aspect keyword to be passed to imshow (default: None)
    cax: Axes, optional
        Color bar axes if necessary (default: None).

    Examples
    --------

    """
    if scale == "log" or scale == "log10":
        norm = ImageNormalize(data, interval=PercentileInterval(100), stretch=LogStretch())
        im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm, aspect=aspect)
    else:
        norm = ImageNormalize(data, interval=PercentileInterval(100))
        im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm, aspect=aspect)

    #im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.grid(color='silver', ls='solid')
    ax.grid(True)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    cb = plt.colorbar(im, ax=ax, cax=cax)
    cb.formatter.set_powerlimits((0, 0))
    cb.locator = MaxNLocator(7, prune=None)
    cb.update_ticks()
    cb.set_label('%s (%s scale)' % (units, scale))  # ,fontsize=16)
    if title != "":
        ax.set_title(title)


#--------------------------------------------------------------------------------------------