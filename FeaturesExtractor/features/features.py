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

        if dy!= 0:
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

    # -----------------------------------------------
    def __init__(self, img):
        """

        :param img: Image used to find the feature (produced by canny edge detection)
        """

        self.my_logger = set_logger(self.__class__.__name__)

        self.img          = img
        self.Nx           = self.img.shape[1]
        self.Ny           = self.img.shape[0]


        self.lines        = []
        self.circles      = []

        self.my_logger.info(f'\n\t Create FeatureImage')

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
            print(line_segment,"  l=" ,theline.length)
            self.lines.append (theline)
            index+=1

        nblines=len(self.lines)

        self.my_logger.info(f'\n\tNumber of Hough lines  {nblines} found')

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
            print(index, " ", center_x," ",center_y," ",radius )
            self.circles.append(thecircle)
            index+=1

        nbcircles=len(self.circles)
        self.my_logger.info(f'\n\tNumber of Hough circles  {nbcircles} found')