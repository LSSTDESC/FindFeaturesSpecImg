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
        norm = ImageNormalize(data, interval=PercentileInterval(98), stretch=LogStretch())
        im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm, aspect=aspect)
    else:
        im = ax.imshow(data, origin='lower', cmap=cmap, aspect=aspect)

    #im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.grid(color='silver', ls='solid')
    ax.grid(True)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    cb = plt.colorbar(im, ax=ax, cax=cax)
    #cb.formatter.set_powerlimits((0, 0))
    #cb.locator = MaxNLocator(7, prune=None)
    #cb.update_ticks()
    cb.set_label('%s (%s scale)' % (units, scale))  # ,fontsize=16)
    if title != "":
        ax.set_title(title)


#--------------------------------------------------------------------------------------------
# https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
#--------------------------------------------------------------------------------------------
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)
#----------------------------------------------------------------------------------------------------


def fit_centralPoint(X1,X2,Y1,Y2,SIGMA=1):
    """

    fit_centralPoint(X1,X2,Y1,Y2)

    find common intersection of all line segments defined by point (X1,Y1) , (X2,Y2)

    :param X1: array of X1 coordinates
    :param X2: array of X2 coordinates
    :param Y1: array of Y1 coordinates
    :param Y2: array of Y2 coordinates
    :param SIGMA  : error on pixels
    :return: x0,y0 : common center (and also errors)
    """

    N=len(X1)
    W=np.sqrt((X2-X1)**2 +(Y2-Y1)**2)   # compute weights
    X21=X2-X1
    Y21=Y2-Y1
    D21=np.sqrt(X21**2+Y21**2)
    ALPHA=X21/D21
    BETA=Y21/D21

    A=(ALPHA*BETA)**2+BETA**4
    B = (ALPHA * BETA)** 2 + ALPHA** 4
    C=ALPHA*BETA**3+ALPHA**3*BETA


    SA   = W/SIGMA**2*A
    SAX  = W/SIGMA**2*A*X1
    SAX2 = W / SIGMA ** 2 * A * X1**2

    SB   = W / SIGMA ** 2 * B
    SBY  = W / SIGMA ** 2 * B * Y1
    SBY2 = W / SIGMA ** 2 * B * Y1 ** 2

    SC   = W / SIGMA **2 * C
    SCX  = W / SIGMA **2 * C * X1
    SCY  = W / SIGMA ** 2 * C * Y1
    SCXY = W / SIGMA ** 2 * C * X1 * Y1


    sa   = np.sum(SA)
    sax  = np.sum(SAX)
    sax2 = np.sum(SAX2)

    sb   = np.sum(SB)
    sby  = np.sum(SBY)
    sby2 = np.sum(SBY2)

    sc   = np.sum(SC)
    scx  = np.sum(SCX)
    scy  = np.sum(SCY)
    scxy = np.sum(SCXY)


    D=sa*sb-sc**2

    # fitted central position
    X0=1/D*(sax*sb+sby*sc -sb*scy -sc*scx)
    Y0=1/D*(sax*sc+sa*sby-sa*scx-sc*scy)

    # errors
    sigX0 = 0.5/D * sb
    sigY0 = 0.5/D * sa
    covXY = 0.5/D * sc

    return X0,Y0,sigX0,sigY0,covXY

#-----------------------------------------------------------------------------------------------


def fit_centralPoint2(X1,X2,Y1,Y2,W,SIGMA=1):
    """

    fit_centralPoint2(X1,X2,Y1,Y2,W)

    find common intersection of all line segments defined by point (X1,Y1) , (X2,Y2)

    :param X1: array of X1 coordinates
    :param X2: array of X2 coordinates
    :param Y1: array of Y1 coordinates
    :param Y2: array of Y2 coordinates
    :param W  : array of weight
    :return: x0,y0 : common center (and also errors)
    """

    N=len(X1)
    X21=X2-X1
    Y21=Y2-Y1
    D21=np.sqrt(X21**2+Y21**2)
    ALPHA=X21/D21
    BETA=Y21/D21

    A=(ALPHA*BETA)**2+BETA**4
    B = (ALPHA * BETA)** 2 + ALPHA** 4
    C=ALPHA*BETA**3+ALPHA**3*BETA


    SA   = W/SIGMA**2*A
    SAX  = W/SIGMA**2*A*X1
    SAX2 = W / SIGMA ** 2 * A * X1**2

    SB   = W / SIGMA ** 2 * B
    SBY  = W / SIGMA ** 2 * B * Y1
    SBY2 = W / SIGMA ** 2 * B * Y1 ** 2

    SC   = W / SIGMA **2 * C
    SCX  = W / SIGMA **2 * C * X1
    SCY  = W / SIGMA ** 2 * C * Y1
    SCXY = W / SIGMA ** 2 * C * X1 * Y1


    sa   = np.sum(SA)
    sax  = np.sum(SAX)
    sax2 = np.sum(SAX2)

    sb   = np.sum(SB)
    sby  = np.sum(SBY)
    sby2 = np.sum(SBY2)

    sc   = np.sum(SC)
    scx  = np.sum(SCX)
    scy  = np.sum(SCY)
    scxy = np.sum(SCXY)


    D=sa*sb-sc**2

    # fitted central position
    X0=1/D*(sax*sb+sby*sc -sb*scy -sc*scx)
    Y0=1/D*(sax*sc+sa*sby-sa*scx-sc*scy)

    # errors
    sigX0 = 0.5/D * sb
    sigY0 = 0.5/D * sa
    covXY = 0.5/D * sc

    return X0,Y0,sigX0,sigY0,covXY

#-----------------------------------------------------------------------------------------------



def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.


    taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]


    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')


    # cut at the same size (SDC, 2020/01/15)
    y_cut= y[int((window_len - 1) / 2):-int((window_len - 1) / 2)]

    return y_cut

#------------------------------------------------------------------------------------------------------------------------



