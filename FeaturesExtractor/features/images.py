
import pandas as pd
import re




class Image(object):

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