from skimage import feature
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import probabilistic_hough_line
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
import numpy as np


from FeaturesExtractor.config import set_logger
from FeaturesExtractor.tools import *
from FeaturesExtractor import parameters

import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------------------------------------------------
class FeatureLine(object):
    # -----------------------------------------------
    def __init__(self, P0,P1,index):
        """

        Input arguments

        :param P0: Coordinates of the first point
        :param P1: Coordinates of the second points
        :param index : index
        """

        self.my_logger   = set_logger(self.__class__.__name__)
        self.x1          = P0[0]       # X coordinate, first point
        self.y1          = P0[1]       # Y coordinate, first point
        self.x2          = P1[0]       # X coordinate, second point
        self.y2          = P1[1]       # Y coordinate, second point

        self.length      = np.sqrt( (self.x2 -self.x1)**2 +  (self.y2 -self.y1)**2)

        dx=self.x2 - self.x1
        dy = self.y2 - self.y1

        self.angle       = 0

        if dx!= 0:
            self.angle       = np.arctan(dy/dx)*180./np.pi



#-----------------------------------------------------------------------------------------------------------------------
class FeatureCircle(object):
    # -----------------------------------------------
    def __init__(self, x,y,r,index):
        """
       
       :param x: x coordinate of circle center
       :param y: y coordinate of circle center
       :param r: r radius of circles
       :param index : number to refer to the circle

        """

        self.my_logger = set_logger(self.__class__.__name__)

        self.x0           = x
        self.y0           = y
        self.r0           = r
        self.index        = index





#----------------------------------------------------------------------------------------------------------------------
class FeatureImage(object):
    """
    class FeatureImage(object)
    """


    # -----------------------------------------------
    def __init__(self, img):
        """

        :param img: Image used to find the feature (produced by canny edge detection)
        """

        self.my_logger = set_logger(self.__class__.__name__)

        self.img           = img
        self.Nx            = self.img.shape[1]
        self.Ny            = self.img.shape[0]

        # Probabilistic Hough Line Detection
        # These lines are supposed to be the track of the first order
        self.lines         = []


        # Hough Circle Detection to detect first approximately the position of the star (order 0)
        self.circles        = []
        self.signal         = np.array([], dtype=float)      # signal summed inside the circle
        self.numberoflines  = np.array([], dtype=int)        # number of segments crossing the circles
        self.numberofpoints = np.array([], dtype=int)        # number of points from extrapolated lines

        self.my_logger.info(f'\n\t Create FeatureImage')

    # ---------------------------------------------------
    def find_lines(self):
        """
        FeatureImage::find_lines()
        :return:
        """


        all_lines = probabilistic_hough_line(self.img, threshold=parameters.LINE_THRESHOLD, line_length=parameters.LINE_LENGTH,
                                                line_gap=parameters.LINE_GAP)


        index=0
        self.lines=[]
        for line_segment in all_lines:
            p0, p1 = line_segment
            theline=FeatureLine(p0,p1,index)
            #print(line_segment,"  l=" ,theline.length)
            self.lines.append (theline)
            index+=1

        nblines=len(self.lines)

        self.my_logger.info(f'\n\tNumber of Hough lines  {nblines} found')

    #--------------------------------------
    def plot_lines(self,img=None, ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                   figsize=[8.3, 7], aspect=None, vmin=None, vmax=None,
                   cmap="jet", cax=None, linecolor="magenta",linewidth=0.5):
        """
        FeatureImage::plot_lines(img)

        :param img:   Image over which the line segment are drown
        :param ax:
        :param scale:
        :param title:
        :param units:
        :param plot_stats:
        :param figsize:
        :param aspect:
        :param vmin:
        :param vmax:
        :param cmap:
        :param cax:
        :return:
        """

        #if img==None:
        if isinstance(img, np.ndarray):
            data=img
        else:
            data=self.img


        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax, aspect=aspect, vmin=vmin,
                          vmax=vmax, cmap=cmap)


        for segm in self.lines:
            plt.plot([segm.x1,segm.x2],[segm.y1,segm.y2],color=linecolor,lw=linewidth)



        # plt.legend()
        if parameters.DISPLAY:
            plt.show()



    #------------------------------------------------------

    def find_circles(self):
        """
        FeatureImage::find_circules()


        :return:
        """
        hough_radii = np.arange(parameters.HOUGH_RADIUS_MIN,parameters.HOUGH_RADIUS_MAX,parameters.HOUGH_RADIUS_STEP)

        hough_res= hough_circle(self.img, hough_radii)

        # Select the most prominent 3 circles
        accums, cx, cy, radii= hough_circle_peaks(hough_res, hough_radii, total_num_peaks=parameters.NB_HOUGH_CIRCLE_PEAKS)

        index=0
        self.circles=[]
        for center_y, center_x, radius in zip(cy, cx, radii):
            thecircle=FeatureCircle(center_x,center_y,radius,index)
            #print(index, " ", center_x," ",center_y," ",radius )
            self.circles.append(thecircle)
            index+=1

        nbcircles=len(self.circles)
        self.my_logger.info(f'\n\tNumber of Hough circles  {nbcircles} found')

    # --------------------------------------
    def plot_circles(self, img=None, ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                       figsize=[8.3, 7], aspect=None, vmin=None, vmax=None,
                       cmap="jet", cax=None, linecolor="magenta", linewidth=1):
        """
            FeatureImage::plot_lines(img)

            :param img:   Image over which the line segment are drown
            :param ax:
            :param scale:
            :param title:
            :param units:
            :param plot_stats:
            :param figsize:
            :param aspect:
            :param vmin:
            :param vmax:
            :param cmap:
            :param cax:
            :return:
            """

        # if img==None:
        if isinstance(img, np.ndarray):
            data = img
        else:
            data = self.img

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax, aspect=aspect, vmin=vmin,
                              vmax=vmax, cmap=cmap)


        for circle in self.circles:
            thecircle = plt.Circle((circle.x0, circle.y0), circle.r0, color=linecolor, fill=False, lw=linewidth)
            ax.add_artist(thecircle)

        # plt.legend()
        if parameters.DISPLAY:
            plt.show()


    def compute_signal_in_circles(self,img):
        """
        FeatureImage::ompute_signal_in_circles(img)

        :param img:  Compute the sum in the image
        :return:
        """

        X = np.arange(0, img.shape[1])
        Y = np.arange(0, img.shape[0])

        XX, YY = np.meshgrid(X, Y)

        for circle in self.circles:
            r = (XX - circle.x0) ** 2 + (YY - circle.y0) ** 2
            signal_in_circle = np.where(r < 9 * circle.r0  ** 2, img, 0)

            sum_in_circle = signal_in_circle.sum()

            sum_in_circle=round(sum_in_circle, 0)
            self.signal=np.append(self.signal,sum_in_circle)

        self.signal=self.signal/self.signal.max()

        return self.signal

