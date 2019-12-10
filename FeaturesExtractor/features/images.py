
import pandas as pd
import re

from FeaturesExtractor.config import set_logger
from FeaturesExtractor.tools import *
from FeaturesExtractor import parameters

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------
class Image(object):

    # -----------------------------------------------
    def __init__(self, file_name):
        """
        The image class contains all the features necessary to load an image and extract a spectrum.

        Parameters
        ----------
        file_name: str
            The file name where the image is.
        target: str, optional
            The target name, to be found in data bases.
        disperser_label: str, optional
            The disperser label to load its properties

        Examples
        --------
        >>> im = Image('tests/data/reduc_20170605_028.fits')
        >>> assert im.file_name == 'tests/data/reduc_20170605_028.fits'
        >>> assert im.data is not None and np.mean(im.data) > 0
        >>> assert im.stat_errors is not None and np.mean(im.stat_errors) > 0
        >>> assert im.header is not None
        >>> assert im.gain is not None and np.mean(im.gain) > 0
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.file_name = file_name

        # container for image processed of not
        self.header         = None
        self.img            = None
        self.theta          = None
        self.lambda_plus    = None
        self.lambda_minus   = None

        # container for clipped images
        self.img_clip          = None
        self.theta_clip        = None
        self.lambda_plus_clip  = None
        self.lambda_minus_clip = None

        # container for edges images
        self.lambda_plus_edges  = None
        self.lambda_minus_edges = None

        self.load_image(file_name)

    # -----------------------------------------------
    def load_image(self, file_name):
        """
        Load the image and store some information from header in class attributes.
        Then load the target and disperser properties. Called when an Image instance is created.

        Parameters
        ----------
        file_name: str
            The fits file name.

        """
        self.header,self.img=load_fits(file_name)
        self.my_logger.info(f'\n\tImage in file  {file_name} loaded')


    # -----------------------------------------------
    def plot_image(self, img_type="img",ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                   figsize=[9.3, 8], aspect=None, vmin=None, vmax=None,
                   cmap="jet", cax=None):
        """Plot image.

        Parameters
        ----------
        ax: Axes, optional
            Axes instance (default: None).
        scale: str
            Scaling of the image (choose between: lin, log or log10) (default: lin)
        title: str
            Title of the image (default: "")
        units: str
            Units of the image to be written in the color bar label (default: "Image units")
        cmap: colormap
            Color map label (default: None)
        vmin: float
            Minimum value of the image (default: None)
        vmax: float
            Maximum value of the image (default: None)
        aspect: str
            Aspect keyword to be passed to imshow (default: None)
        cax: Axes, optional
            Color bar axes if necessary (default: None).
        figsize: tuple
            Figure size (default: [9.3, 8]).
        plot_stats: bool
            If True, plot the uncertainty map instead of the image (default: False).

        Examples
        --------

        """
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        if img_type=="img":
            data = np.copy(self.img)
        elif img_type=="lambda_p":
            data = np.copy(self.lambda_plus)
        elif img_type=="lambda_m":
            data = np.copy(self.lambda_minus)
        elif img_type=="theta":
            data = np.copy(self.theta)


        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax,aspect=aspect, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.legend()
        if parameters.DISPLAY:
            plt.show()


    #--------------------------------------------------------------------------------------------

    def process_image(self):
        """

        :return:
        """
        self.lambda_plus, self.lambda_minus, self.theta=hessian_and_theta(self.img)

        self.my_logger.info(f'\n\tImage processed')

    # --------------------------------------------------------------------------------------------


