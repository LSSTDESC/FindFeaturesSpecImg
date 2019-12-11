from numpy.core._multiarray_umath import ndarray
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

        self.angle          = 0           # angle of segment in degree

        self.flag           = False       # this segment will be validated if it is in a validated circle
        self.nbcircles      = 0
        self.nbpixincircles = 0

        if dx!= 0:
            self.angle         = np.arctan(dy/dx)*180./np.pi




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
        self.flag_validated_circles = np.array([], dtype=bool)     # number of validated circle

        self.my_logger.info(f'\n\t Create FeatureImage')

    # ---------------------------------------------------------------
    def find_lines(self):
        """
        FeatureImage::find_lines()
        :return:
        """
        self.my_logger.info(f'\n\t probabilistic Hough Line search')

        all_lines = probabilistic_hough_line(self.img, threshold=parameters.LINE_THRESHOLD, line_length=parameters.LINE_LENGTH,
                                                line_gap=parameters.LINE_GAP)


        index=0
        self.lines=[]
        for line_segment in all_lines:
            p0, p1 = line_segment

            dx=p0[0]-p1[0]
            dy = p0[1] - p1[1]

            if dx!=0 and dy!=0 :
                theline=FeatureLine(p0,p1,index)
                #print(line_segment,"  l=" ,theline.length)
                self.lines.append (theline)
                index+=1

        nblines=len(self.lines)

        self.my_logger.info(f'\n\tNumber of Hough lines  {nblines} found')

    #--------------------------------------
    def plot_lines(self,img=None, ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                   figsize=[7.5, 7], aspect=None, vmin=None, vmax=None,
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

        self.my_logger.info(f'\n\t plot lines')

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



    #-----------------------------------------------------------------

    def find_circles(self):
        """
        FeatureImage::find_circules()


        :return:
        """

        self.my_logger.info(f'\n\t Hough circles search')


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

    # ------------------------------------------------------------
    def plot_circles(self, img=None, ax=None, scale="lin", title="", units="Image units", plot_stats=False,
                       figsize=[7.5, 7], aspect=None, vmin=None, vmax=None,
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

        self.my_logger.info(f'\n\t plot circles ')

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



        index=0
        for circle in self.circles:
            # if the validation of circles has proceed
            if len(self.flag_validated_circles)>0:
                if self.flag_validated_circles[index]:
                    thecircle = plt.Circle((circle.x0, circle.y0), circle.r0, color=linecolor, fill=False, lw=linewidth)
                    ax.add_artist(thecircle)
            # the validation of circles has not proceed
            else:
                thecircle = plt.Circle((circle.x0, circle.y0), circle.r0, color=linecolor, fill=False, lw=linewidth)
                ax.add_artist(thecircle)
            index+=1  # increase at each new circle

        # plt.legend()
        if parameters.DISPLAY:
            plt.show()


    #----------------------------------------------------------------

    def compute_signal_in_circles(self,img):
        """
        FeatureImage::compute_signal_in_circles(img)

        :param img:  Compute the sum in the image
        :return:
        """

        self.my_logger.info(f'\n\t compute signal in circles')

        X = np.arange(0, img.shape[1])
        Y = np.arange(0, img.shape[0])

        XX, YY = np.meshgrid(X, Y)

        self.signal = np.array([], dtype=float)

        if len(self.circles)==0:
            return self.signal

        # loop in circles
        for circle in self.circles:
            r = (XX - circle.x0) ** 2 + (YY - circle.y0) ** 2
            signal_in_circle = np.where(r < 9 * circle.r0  ** 2, img, 0)

            sum_in_circle = signal_in_circle.sum()

            sum_in_circle=round(sum_in_circle, 0)
            self.signal=np.append(self.signal,sum_in_circle)

        self.signal=self.signal/self.signal.max()

        return self.signal

    # --------------------------------------------------------------------------
    def compute_line_in_circles(self):
        """
        FeatureImage::compute_line_in_circles(self)

        Compute the number of lines and pixels in the circles

        :return:
        """

        self.my_logger.info(f'\n\t compute line in circles ')

        X = np.arange(0, self.Nx)

        self.numberoflines = np.array([], dtype=int)   # number of segments crossing the circles
        self.numberofpoints = np.array([], dtype=int)  # number of points from extrapolated lines

        # loop in circles
        for circle in self.circles:
            # loop on lines
            NumberOfCrossings = 0
            NumberOfPixels    = 0
            # loop on lines
            for line in self.lines:
                # make a straight lines
                Z   = np.polyfit([line.x1,line.x2], [line.y1,line.y2], 1)
                pol = np.poly1d(Z)
                Y   = pol(X)
                theindexes=np.where( (X-circle.x0)**2+(Y-circle.y0)**2 <circle.r0**2)[0]
                if len(theindexes)>0:
                    NumberOfCrossings+=1
                    NumberOfPixels+=len(theindexes)

            # compute circle by circles
            self.numberoflines  =  np.append(self.numberoflines, NumberOfCrossings)
            self.numberofpoints =  np.append(self.numberofpoints, NumberOfPixels)

        return self.numberoflines,self.numberofpoints

    # --------------------------------------------------------------------------
    def flag_validate_circles(self):
        """
        FeatureImage::flag_validate_circles()

        Function to validate a circle if it is crossed by enough segments-pixels and has enough signal in it

        :return:
        """

        self.my_logger.info(f'\n\t flag_validate circles')

        # reset
        self.flag_validated_circles = np.array([], dtype=bool)

        # by default plot every circles
        self.flag_validated_circles = np.full(shape=len(self.circles), fill_value=True, dtype=np.bool)



        if len(self.numberoflines) > 0 and len(self.numberofpoints) > 0:
            erase_index1 = np.where(np.logical_or(self.numberofpoints == 0, self.numberoflines == 0))[0]
            print("erase_index1 = ", erase_index1)
            print("self.signal = ", self.signal)
            erase_index2 = np.where(self.signal < parameters.HOUGH_SIGNAL_THRESHOLD)[0]
            print("erase_index2 = ", erase_index2)
            erase_index = np.union1d(erase_index1, erase_index2)
            print("erase_index = ", erase_index)
            self.flag_validated_circles[erase_index] = False

        print("flag_plot_circle = ", self.flag_validated_circles)

        index=0
        for index in np.arange(len(self.flag_validated_circles)):
            x0=self.circles[index].x0
            y0 = self.circles[index].y0
            r0 = self.circles[index].r0
            print("{} :: ({},{}) {}".format(index,x0,y0,r0))

    #----------------------------------------------------------------------------------
    def flag_validate_lines(self):
        """
        FeatureImage::flag_validate_lines(self)

        Function to validate if a segment is associated to a validated circle.
        (Will allow to caculate the angle theta)

        :return:
        """

        self.my_logger.info(f'\n\t flag validate lines')

        print("self.flag_validated_circles = ", self.flag_validated_circles)

        index = 0
        X = np.arange(0, self.Nx)

        # loop on circles
        for circle in self.circles:
            # if the validation of circles has proceed
            if len(self.flag_validated_circles) > 0:
                if self.flag_validated_circles[index]:
                    # loop on lines
                    for line in self.lines:
                        # make a straight lines
                        Z = np.polyfit([line.x1, line.x2], [line.y1, line.y2], 1)
                        pol = np.poly1d(Z)
                        Y = pol(X)
                        theindexes = np.where((X - circle.x0) ** 2 + (Y - circle.y0) ** 2 < circle.r0 ** 2)[0]
                        if len(theindexes)>0:
                            line.flag=True
                            line.nbcircles += 1
                            line.nbpixincircles += len(theindexes)
    #------------------------------------------------------------------------------------
    def plot_circles_profiles(self):
        """

        FeatureImage::plot_circles_profiles(self

        :return:
        """

        self.my_logger.info(f'\n\t plot circles profiles')


        n1 = 3.0
        n2 = 2.0
        n3 = 1.0
        n4 = 4.0

        print("self.flag_validated_circles = ",self.flag_validated_circles)

        index=0
        # loop on circles
        for circle in self.circles:
            # if the validation of circles has proceed
            if len(self.flag_validated_circles) > 0:
                if self.flag_validated_circles[index]:
                    y0=circle.y0
                    x0=circle.x0
                    r0=circle.r0


                    bandX = self.img[int(y0 - n2 * r0):int(y0 + n2 * r0), int(x0 - n2 * r0):int(x0 + n2 * r0)]
                    bandY = self.img[int(y0 - n2 * r0):int(y0 + n2 * r0), int(x0 - n2 * r0):int(x0 + n2 * r0)]

                    profX = np.sum(bandX, axis=0)
                    profY = np.sum(bandY, axis=1)

                    xx = np.arange(int(x0 - n2 * r0), int(x0 + n2 * r0))
                    yy = np.arange(int(y0 - n2 * r0), int(y0 + n2 * r0))


                    title="circle (x0,y0) = ({},{}) , r0 = {}".format(x0,y0,r0)
                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

                    ax1.plot(xx,profX,"r-")
                    ax1.grid()
                    ax1.set_xlabel("X")
                    ax1.set_title("X")

                    ax2.plot(yy, profY, "r-")
                    ax2.grid()
                    ax2.set_xlabel("Y")
                    ax2.set_title("Y")

                    plt.suptitle(title)
                    plt.show()
            index+=1




