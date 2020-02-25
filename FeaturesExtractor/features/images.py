
import pandas as pd
import re

from FeaturesExtractor.config import set_logger
from FeaturesExtractor.tools import *
from FeaturesExtractor import parameters

import matplotlib.pyplot as plt

from skimage import feature

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
        self.header            = None
        self.Nx                 = 0
        self.Ny                 = 0
        self.img_cube           = None

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
        #self.header,self.img=load_fits(file_name)
        header,img = load_fits(file_name)


        self.Nx = img.shape[1]
        self.Ny = img.shape[0]

        self.img_cube = np.zeros((parameters.NBIMG,self.Ny,self.Nx))
        self.img_cube[parameters.IndexImg.img,:,:] = img


        self.my_logger.info(f'\n\tImage in file  {file_name} loaded')


    # -----------------------------------------------
    def plot_image(self, img_type="img",ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                   figsize=[8.3, 7], aspect=None, vmin=None, vmax=None,
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
            data = np.copy(self.img_cube[parameters.IndexImg.img])
        elif img_type=="lambda_p":
            data = np.copy(self.img_cube[parameters.IndexImg.lambda_plus])
        elif img_type=="lambda_m":
            data = np.copy(self.img_cube[parameters.IndexImg.lambda_minus])
        elif img_type=="theta":
            data = np.copy(self.img_cube[parameters.IndexImg.theta])
        elif img_type=="img_cut":
            data = np.copy(self.img_cube[parameters.IndexImg.img_clip])
        elif img_type=="lambda_p_cut":
            data = np.copy(self.img_cube[parameters.IndexImg.lambda_plus_clip])
        elif img_type=="lambda_m_cut":
            data = np.copy(self.img_cube[parameters.IndexImg.lambda_minus_clip])
        elif img_type=="theta_cut":
            data = np.copy(self.img_cube[parameters.IndexImg.theta_clip])
        elif img_type == "lambda_p_edge":
            data = np.copy(self.img_cube[parameters.IndexImg.lambda_plus_edges])
        elif img_type == "lambda_m_edge":
            data = np.copy(self.img_cube[parameters.IndexImg.lambda_minus_edges])


        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax, aspect=aspect, vmin=vmin,
                              vmax=vmax, cmap=cmap)

        #plt.legend()
        if parameters.DISPLAY:
            plt.show()

    #----------------------------------------------------------------------------------------------------------------

    def plot_edges(self):
        """

        :return:
        """

        # display results
        #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharex=True, sharey=True)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))


        ax1.imshow(self.img_cube[parameters.IndexImg.lambda_plus_edges], origin="lower", cmap=plt.cm.gray)
        ax1.set_title('Canny filter on lambda_plus, $\sigma=${}'.format(parameters.SIGMA_EDGE), fontsize=14)
        ax1.grid(True)
        ax1.set_xlabel('X [pixels]')
        ax1.set_ylabel('Y [pixels]')


        ax2.imshow(self.img_cube[parameters.IndexImg.lambda_minus_edges], origin="lower", cmap=plt.cm.gray)
        ax2.set_title('Canny filter on lambda_minus, $\sigma=${}'.format(parameters.SIGMA_EDGE), fontsize=14)
        ax2.grid(True)
        ax2.set_xlabel('X [pixels]')
        ax2.set_ylabel('Y [pixels]')

        # fig.tight_layout()
        plt.suptitle("Canny edge detection",fontsize=18)

        plt.show()

    #--------------------------------------------------------------------------------------------
    def process_image(self):
        """

        :return:
        """


        lambda_plus, lambda_minus, theta=hessian_and_theta(self.img_cube[parameters.IndexImg.img,:,:],margin_cut=1)

        lambda_plus_cut = -np.where(lambda_plus < 0, lambda_plus, 0)


        Nx=lambda_plus.shape[1]
        Ny=lambda_plus.shape[0]

        self.my_logger.info(f'\n\tSize of original image (Nx,Ny) = ({self.Nx},{self.Ny})')
        self.my_logger.info(f'\n\tSize of processed image (Nx,Ny) = ({Nx},{Ny})')

        Xpad = int((self.Nx - Nx) / 2)
        Ypad = int((self.Ny - Ny) / 2)

        self.my_logger.info(f'\n\tPadding of processed image (Xpad,Ypad)  = ({Xpad},{Ypad})')

        # copy the processed image in the cube
        self.img_cube[parameters.IndexImg.lambda_plus,Ypad:self.Ny - Ypad, Xpad:self.Nx - Xpad]     = lambda_plus
        #self.img_cube[parameters.IndexImg.lambda_plus, Ypad:self.Ny - Ypad, Xpad:self.Nx - Xpad]    = lambda_plus_cut


        self.img_cube[parameters.IndexImg.lambda_minus,Ypad:self.Ny - Ypad, Xpad:self.Nx - Xpad]    = lambda_minus
        self.img_cube[parameters.IndexImg.theta,Ypad:self.Ny - Ypad, Xpad:self.Nx - Xpad]           = theta


        self.my_logger.info(f'\n\tImage processed')

    # --------------------------------------------------------------------------------------------
    def clip_images(self):
        """

        :return:
        """

        for N in np.arange(4):
            thearray=self.img_cube[N]
            theclippedarray=clip_array(thearray,parameters.CLIP_MIN,parameters.CLIP_MAX)
            self.img_cube[N+4] = theclippedarray



        self.my_logger.info(f'\n\tImages clipped')

    #------------------------------------------------------------------------------------------
    def compute_edges(self):
        """

        :return:
        """

        self.img_cube[parameters.IndexImg.lambda_minus_edges] = \
            feature.canny(self.img_cube[parameters.IndexImg.lambda_minus_clip], sigma=parameters.SIGMA_EDGE)
        self.img_cube[parameters.IndexImg.lambda_plus_edges] = \
            feature.canny(self.img_cube[parameters.IndexImg.lambda_plus_clip], sigma=parameters.SIGMA_EDGE)

        self.my_logger.info(f'\n\tImages edges computed')