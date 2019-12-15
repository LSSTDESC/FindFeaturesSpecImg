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
from matplotlib.patches import Circle
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

from astropy.table import Table


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
        self.nbcircles      = 0           # number of circles associated to that line
        self.nbpixincircles = 0           # number of pixels from the extrapolated line in the circle

        self.circlesindex   = np.array([], dtype=int)   # container of associated circle

        if dx!= 0:
            self.angle         = np.arctan(dy/dx)*180./np.pi   # angle of that segment




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

        self.x_fit        = 0
        self.y_fit        = 0


    def distance(self,acircle):
        return np.sqrt((acircle.x0-self.x0)**2 + (acircle.y0-self.y0)**2 )



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

        self.img           = img                              # base edge image on which circles and line are found
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
        self.flag_saturation_circles = np.array([], dtype=bool)

        self.circlesummary        = Table(names=('index', 'x0', 'y0' ,"r0"), dtype=('i4', 'i4','i4','i4'))

        self.my_logger.info(f'\n\t Create FeatureImage')

    # ----------------------------------------------------------------------------------------------------------------
    def find_lines(self):
        """
        FeatureImage::find_lines()
        :return:
        """
        #-----------------------------------------------------------------------------------------------------------
        self.my_logger.info(f'\n\t probabilistic Hough Line search')

        all_lines = probabilistic_hough_line(self.img, threshold=parameters.LINE_THRESHOLD, line_length=parameters.LINE_LENGTH,
                                                line_gap=parameters.LINE_GAP)


        index=0
        self.lines=[]
        for line_segment in all_lines:
            p0, p1 = line_segment

            dx=p1[0]-p0[0]
            dy = p1[1] - p0[1]

            if dx!=0 and dy!=0 :
                if dx > 0:
                    theline=FeatureLine(p0,p1,index)
                else:
                    theline = FeatureLine(p1, p0, index)

                self.lines.append (theline)
                index+=1

        nblines=len(self.lines)

        self.my_logger.info(f'\n\tNumber of Hough lines  {nblines} found')

    #-------------------------------------------------------------------------------------------------------------------
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
        #-------------------------------------------------------------------------------------------------------------

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



    #-------------------------------------------------------------------------------------------------------------------

    def erase_lines(self,img=None):
        """
        FeatureImage::erase_lines(img)

         Erase lines in the image
         If img=None Erase line in canny edge before applying Hough Circle algorithm

        :param img : input image on which to erase the lines. If not provided None, the lines are erased in the canny edge
        :return:   : image on which the line erase has been done
        """
        #---------------------------------------------------------------------------------------------------------------

        self.my_logger.info(f'\n\t erase lines')

        # if img==None:
        if isinstance(img, np.ndarray):
            data = img
            title = "Erase lines in the provided image"
        else:
            data = self.img
            title = "Erased lines in canny image"

        X = np.arange(0, self.img.shape[1])
        Y = np.arange(0, self.img.shape[0])

        XX, YY = np.meshgrid(X, Y)


        # loop on line segement
        index=0
        for segm in self.lines:
            x1=segm.x1
            x2=segm.x2
            y1=segm.y1
            y2=segm.y2
            dx=x2-x1
            dy=y2-y1

            selected_elements  = np.logical_and(
                np.logical_and(XX>=x1,XX<=x2),
                np.logical_and(YY >= dy/dx*(XX-x1)+y1-parameters.LINE_ERASE_MARGIN,
                                                        YY <= dy/dx*(XX-x1)+y1+parameters.LINE_ERASE_MARGIN))

            #if(index==0):
            #    newimg=np.zeros(self.img.shape,dtype=float)
            #    newimg[selected_elements]=1
            #    plt.figure(figsize=(5,5))
            #    plt.imshow(newimg,origin="lower",cmap="gray")
            #    plt.show()

            #self.img[selected_elements]=0.0
            data[selected_elements] = 0.0

            index+=1
        #end of loop on segment

        plt.figure(figsize=(7,7))
        ax = plt.gca()

        #plot_image_simple(ax, data=self.img, scale="lin", title="Erased lines in canny image",cmap="gray")
        plot_image_simple(ax, data=data, scale="lin", title=title, cmap="gray")

        # plt.legend()
        if parameters.DISPLAY:
            plt.show()

        return data



    #--------------------------------------------------------------------------------------------------------------------

    def find_circles(self):
        """
        FeatureImage::find_circules()


        :return:
        """
        #-----------------------------------------------------------------------------------------------------------------

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
            self.circlesummary.add_row((index,center_x,center_y,radius))
            index+=1

        if parameters.DEBUG:
            print(self.circlesummary)

        nbcircles=len(self.circles)
        self.my_logger.info(f'\n\tNumber of Hough circles  {nbcircles} found')

    # -----------------------------------------------------------------------------------------------------------------
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
        #---------------------------------------------------------------------------------------------------------------

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


    #--------------------------------------------------------------------------------------------------------------------

    def compute_signal_in_circles(self,img):
        """
        FeatureImage::compute_signal_in_circles(img)

        :param img:  Compute the sum in the image
        :return:
        """
        #------------------------------------------------------------------------------------------------------------------

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

        self.circlesummary["signal"]  =  self.signal
        self.signal=self.signal/self.signal.max()
        self.circlesummary['signal'].format = "%10.0f"

        self.circlesummary["fraction"] = self.signal
        self.circlesummary['fraction'].format = "%1.3f"

        if parameters.DEBUG:
            print(self.circlesummary)


        return self.signal

    # ------------------------------------------------------------------------------------------------------------------
    def compute_line_in_circles(self,img=None,ax=None, scale="log", title="Extrapolated lines", units="Image units", plot_stats=False,
                       figsize=[7.5, 7], aspect=None, vmin=None, vmax=None,
                       cmap="gray", cax=None, linecolor="magenta", linewidth=0.5):
        """
        FeatureImage::compute_line_in_circles(self)

        Compute the number of lines and pixels in the circles

        :return:
        """
        #----------------------------------------------------------------------------------------------------------------
        self.my_logger.info(f'\n\t compute line in circles ')


        if isinstance(img, np.ndarray):
            data = img
        else:
            data = self.img


        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()



        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax, aspect=aspect, vmin=vmin,
                          vmax=vmax, cmap=cmap)


        X = np.arange(0, data.shape[1])

        self.numberoflines  = np.array([], dtype=int)   # number of segments crossing the circles
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
                theindexes = np.where( (X-circle.x0)**2+(Y-circle.y0)**2 <circle.r0**2)[0]
                if len(theindexes) > 0:
                    NumberOfCrossings+=1
                    NumberOfPixels+=len(theindexes)
                    ax.plot(X,Y,'r-',lw=0.5)

            # compute circle by circles
            self.numberoflines  =  np.append(self.numberoflines, NumberOfCrossings)
            self.numberofpoints =  np.append(self.numberofpoints, NumberOfPixels)

        self.circlesummary["NbLines"]  = self.numberoflines
        self.circlesummary["NbPoints"] = self.numberofpoints

        if parameters.DEBUG:
            print(self.circlesummary)

        plt.xlim(0,data.shape[1])
        plt.ylim(0,data.shape[0])
        plt.show()


        return self.numberoflines,self.numberofpoints

    # -----------------------------------------------------------------------------------------------------------------
    def flag_validate_circles(self):
        """
        FeatureImage::flag_validate_circles()

        Function to validate a circle if it is crossed by enough segments-pixels and has enough signal in it

        :return:
        """
        #---------------------------------------------------------------------------------------------------------------

        self.my_logger.info(f'\n\t flag_validate circles')

        # reset
        self.flag_validated_circles = np.array([], dtype=bool)

        # by default plot every circles
        self.flag_validated_circles = np.full(shape=len(self.circles), fill_value=True, dtype=np.bool)


        # Step One : Erase irrelevant circles
        if len(self.numberoflines) > 0 and len(self.numberofpoints) > 0:
            erase_index1 = np.where(np.logical_or(self.numberofpoints == 0, self.numberoflines == 0))[0]
            erase_index2 = np.where(self.signal < parameters.CIRCLE_SIGNAL_THRESHOLD)[0]
            erase_index = np.union1d(erase_index1, erase_index2)
            self.flag_validated_circles[erase_index] = False



        # Step Tow : Eliminate encapsulated circles
        all_pairs=[]
        remaining_ids=np.where(self.flag_validated_circles)[0]


        for idx1 in np.arange(len(remaining_ids)):
            id1=remaining_ids[idx1]
            for idx2 in np.arange(idx1+1,len(remaining_ids)):
                id2=remaining_ids[idx2]
                dist=self.circles[id1].distance(self.circles[id2])
                # make a circle pair
                if dist <= parameters.CIRCLE_MIN_DISTANCE :
                    all_pairs.append((id1,id2))

        # erase the circle with the smallest radius
        for p in all_pairs:
            id1,id2=p
            if self.circles[id1].r0 < self.circles[id2].r0:
                print("unvalidate circle ",id1)
                self.flag_validated_circles[id1] = False
                self.circlesummary["validation"] = self.flag_validated_circles[id1]
            else:
                print("unvalidate circle", id2)
                self.flag_validated_circles[id2] = False
                self.circlesummary["validation"] = self.flag_validated_circles[id2]

        # copy the validation flag in circle summary
        self.circlesummary["validation"] = self.flag_validated_circles

        if parameters.DEBUG:
            print(self.circlesummary)



    #------------------------------------------------------------------------------------------------------------------
    def flag_validate_lines(self):
        """
        FeatureImage::flag_validate_lines(self)

        Function to validate if a segment is associated to a validated circle.
        (Will allow to caculate the angle theta)

        :return:
        """

        self.my_logger.info(f'\n\t flag validate lines')



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
                            line.circlesindex=np.append(line.circlesindex,circle.index)  # add in line object a reference to the circle
            index+=1 # increase

    #------------------------------------------------------------------------------------------------------------------------
    def plot_validated_lines(self,img=None,ax=None, scale="log", title="Validated lines", units="Image units", plot_stats=False,
                       figsize=[7.5, 7], aspect=None, vmin=None, vmax=None,
                       cmap="gray", cax=None, linecolor="magenta", linewidth=0.5):
        """

        :param mg:
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
        :param linecolor:
        :param linewidth:
        :return:
        """

        self.my_logger.info(f'\n\t plot validated lines ')


        #mycol=["r","b","g","m","orange","y","c", "r","b","g","m","orange","y","c"]

        #discretized_jet = cmap_discretize(matplotlib.cm.jet, len(self.circles))

        # wavelength bin colors
        jet = plt.get_cmap('jet')
        cNorm = mpl.colors.Normalize(vmin=0, vmax=len(self.circles))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        all_colors = scalarMap.to_rgba(np.arange(len(self.circles)), alpha=1)

        if isinstance(img, np.ndarray):
            data = img
        else:
            data = self.img

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        plot_image_simple(ax, data=data, scale=scale, title=title, units=units, cax=cax, aspect=aspect, vmin=vmin,
                          vmax=vmax, cmap=cmap)

        # loop on lines
        for segm in self.lines:
            if segm.flag:
                #col=mycol[segm.circlesindex[0]]
                col=all_colors[segm.circlesindex[0]]
                ax.plot([segm.x1,segm.x2],[segm.y1,segm.y2],color=col,lw=linewidth)

        # loop on circles
        idx=0
        for circle in self.circles:
            if self.flag_validated_circles[idx]:
                #col = mycol[idx]
                col = all_colors[idx]
                thecircle = Circle((circle.x0, circle.y0), circle.r0, color=col, fill=False, lw=2)

                ax.add_patch(thecircle)

            idx+=1




        plt.show()
    #-----------------------------------------------------------------------------------------------
    def compute_theta(self):
        """

        :return:
        """

        theta_table = np.zeros(len(self.circles))
        err_theta_table = np.zeros(len(self.circles))

        all_theta_circle =[]
        all_weight_circle  = []

        idx = 0
        # loop on circle
        for circle in self.circles:
            all_theta_circle = []
            all_weight_circle = []
            if self.flag_validated_circles[idx]:

                # loop on line
                for segm in self.lines:
                    if segm.flag:
                        if segm.circlesindex[0] == idx and segm.length>0 :
                            all_theta_circle.append(segm.angle)
                            all_weight_circle.append(segm.length)

                all_theta_circle=np.array(all_theta_circle)
                all_weight_circle=np.array(all_weight_circle)
                weight_sum=np.sum(all_weight_circle)

                if weight_sum > 0 :
                    themean,thesigma=weighted_avg_and_std(all_theta_circle, all_weight_circle)
                else:
                    themean  = 0
                    thesigma = 0

                theta_table[idx]=themean
                err_theta_table[idx]=thesigma


            idx += 1

        self.circlesummary["theta_mean"] = theta_table
        self.circlesummary["theta_rms"]  = err_theta_table

        self.circlesummary["theta_mean"].format = "%3.2f"
        self.circlesummary["theta_rms"].format = "%3.2f"

        if parameters.DEBUG:
            print(self.circlesummary)



    #------------------------------------------------------------------------------------------------
    def get_circles_inprofiles(self,img):
        """

        FeatureImage::get_circles_inprofiles

        For each validated circle, it extract the profile at get the true center

        :return:
        """

        self.my_logger.info(f'\n\t circles profiles')


        n1 = parameters.NBRADIUS
        n2 = parameters.RADIUSFRACTION

        fit_X0=np.zeros(len(self.circles))
        fit_Y0 = np.zeros(len(self.circles))

        errfit_X0 = np.zeros(len(self.circles))
        errfit_Y0 = np.zeros(len(self.circles))

        index=0
        # loop on circles
        for circle in self.circles:
            # if the validation of circles has proceed
            if len(self.flag_validated_circles) > 0:
                if self.flag_validated_circles[index]:
                    y0=circle.y0
                    x0=circle.x0
                    r0=circle.r0
                    idx0=circle.index


                    # extract the profile
                    bandX = img[int(y0 - n1 * r0):int(y0 + n1 * r0), int(x0 - n1 * r0):int(x0 + n1 * r0)]
                    bandY = img[int(y0 - n1 * r0):int(y0 + n1 * r0), int(x0 - n1 * r0):int(x0 + n1 * r0)]

                    bandX_cut = img[int(y0 - n1 * r0):int(y0 + n1 * r0), int(x0 - n2 * r0):int(x0 + n2 * r0)]
                    bandY_cut = img[int(y0 - n2 * r0):int(y0 + n2 * r0), int(x0 - n1 * r0):int(x0 + n1 * r0)]

                    profX = np.sum(bandX, axis=0)
                    profY = np.sum(bandY, axis=1)

                    profX_cut = np.sum(bandX_cut, axis=0)
                    profY_cut = np.sum(bandY_cut, axis=1)

                    profXMIN=profX.min()
                    profXMAX = profX.max()

                    profYMIN = profY.min()
                    profYMAX = profY.max()

                    ## get x and y axis
                    xx = np.arange(int(x0 - n1 * r0), int(x0 + n1 * r0))
                    yy = np.arange(int(y0 - n1 * r0), int(y0 + n1 * r0))

                    # left , right, bottom, top
                    extent_BandX = [int(x0 - n1 * r0),int(x0 + n1 * r0), int(y0 - n1 * r0), int(y0 + n1 * r0)  ]
                    extent_BandY = [int(x0 - n1 * r0), int(x0 + n1 * r0), int(y0 - n1 * r0), int(y0 + n1 * r0)]

                    xx_cut = np.arange(int(x0 - n2 * r0), int(x0 + n2 * r0))
                    yy_cut = np.arange(int(y0 - n2 * r0), int(y0 + n2 * r0))

                    ## Fit the profile
                    deg = 5

                    # fit
                    z_x = np.polyfit(xx_cut, profX_cut, deg)
                    z_y = np.polyfit(yy_cut, profY_cut, deg)

                    # poly
                    p_x = np.poly1d(z_x)
                    p_y = np.poly1d(z_y)

                    ##
                    fit_val_profX=p_x(xx_cut)
                    fit_val_profY = p_y(yy_cut)

                    # derivatives of polynoms
                    dp_x = np.polyder(p_x)
                    dp_y = np.polyder(p_y)

                    # roots of the derivatives
                    roots_in_x = np.roots(dp_x)
                    roots_in_y = np.roots(dp_y)

                    # find the realistic root close to circle center
                    idd = np.argmin(np.abs(roots_in_x - x0))
                    the_fit_x = roots_in_x[idd]

                    idd = np.argmin(np.abs(roots_in_y - y0))
                    the_fit_y = roots_in_y[idd]


                    # get the root being real
                    the_fit_x = np.real(the_fit_x)
                    the_fit_y = np.real(the_fit_y)

                    #the error
                    err_x = np.abs(the_fit_x - x0)
                    err_y = np.abs(the_fit_y - y0)

                    fit_X0[idx0]  = the_fit_x
                    fit_Y0[idx0] =  the_fit_y

                    errfit_X0[idx0] = err_x
                    errfit_Y0[idx0] = err_y

                    # save the fit result for later optimisation
                    circle.x_fit    = the_fit_x
                    circle.y_fit    = the_fit_y


                    # plot profile and 2D view for each circle

                    title = "circle  id={} :: (x0,y0) = ({},{}) , r0 = {}".format(idx0, x0, y0, r0)


                    ##  Profile
                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
                    ax1.plot(xx,profX,"r-",label=" $\lambda$ X profile")
                    ax1.plot([x0,x0],[profXMIN,profXMAX],"k-",label="circle X center")
                    ax1.plot([x0-r0, x0-r0], [profXMIN, profXMAX], "k:")
                    ax1.plot([x0 +r0, x0 + r0], [profXMIN, profXMAX], "k:")

                    ax1.plot(xx_cut,fit_val_profX,"g-",label="fitted X profile")
                    ax1.plot([the_fit_x,the_fit_x],[profXMIN,profXMAX],"b-",label="fitted X minimum")


                    ax1.grid()
                    ax1.set_xlabel("X")
                    ax1.set_title("X")
                    ax1.legend()

                    ax2.plot(yy, profY, "r-",label="$\lambda$ Y profile")
                    ax2.plot([y0, y0], [profYMIN, profYMAX], "k-",label="circle Y center")
                    ax2.plot([y0 - r0, y0 - r0], [profYMIN, profYMAX], "k:")
                    ax2.plot([y0 + r0, y0 + r0], [profYMIN, profYMAX], "k:")

                    ax2.plot(yy_cut, fit_val_profY, "g-",label="fitted Y profile")
                    ax2.plot([the_fit_y, the_fit_y], [profXMIN, profXMAX], "b-",label="fitted Y minimum")

                    ax2.grid()
                    ax2.set_xlabel("Y")
                    ax2.set_title("Y")
                    ax2.legend()

                    plt.suptitle(title)
                    plt.show()


                    ## 2D Plot
                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
                    ax1.imshow(bandX, cmap="gray",extent= extent_BandX )
                    thecircle1 = Circle((x0, y0), r0, color="red", fill=False, lw=2)
                    ax1.add_patch(thecircle1)
                    # Draw a point at the location (3, 9) with size 1000
                    ax1.scatter( the_fit_x,  the_fit_y, s=10,color="magenta")
                    ax1.set_title("Band X")
                    ax1.set_xlabel("X")

                    ax2.imshow(bandY, cmap="gray",extent= extent_BandY )
                    thecircle2 = Circle((x0, y0), r0, color="blue", fill=False, lw=2)
                    ax2.add_patch(thecircle2)
                    ax2.scatter(the_fit_x, the_fit_y, s=10, color="magenta")
                    ax2.set_title("Band Y")
                    ax2.set_xlabel("Y")
                    plt.suptitle(title)
                    plt.show()



            index+=1

        # save in summary
        self.circlesummary["x0_fit"] = fit_X0
        self.circlesummary["y0_fit"] = fit_Y0

        self.circlesummary["err_x0_fit"] = errfit_X0
        self.circlesummary["err_y0_fit"] = errfit_Y0

        self.circlesummary["x0_fit"].format = "%3.2f"
        self.circlesummary["y0_fit"].format = "%3.2f"
        self.circlesummary["err_x0_fit"].format = "%3.2f"
        self.circlesummary["err_y0_fit"].format = "%3.2f"

        if parameters.DEBUG:
            print(self.circlesummary)


    #--------------------------------------------------------------------------------------------------------------------------
    def test_incircle_saturation(self,img,title="original image",figsize=[8, 8],cmap="jet"):
        """
         FeatureImage::test_incircle_saturation(img)

         Test if there is some saturation in the image

        :param img:
        :return:
        """

        self.my_logger.info(f'\n\t test if there is some saturation inside circle')

        # reset
        self.flag_saturation_circles = np.array([], dtype=bool)

        # by default plot every circles
        self.flag_saturation_circles = np.full(shape=len(self.circles), fill_value=True, dtype=np.bool)

        w = int(parameters.VIGNETTE_SIZE / 2)

        index = 0
        # loop on circles
        for circle in self.circles:
            # if the validation of circles has proceed
            if len(self.flag_validated_circles) > 0:
                if self.flag_validated_circles[index]:
                    y0 = int(circle.y_fit)
                    x0 = int(circle.x_fit)
                    idx0 = circle.index

                    # additionnal constraint on circle (x,y fit had to be done)
                    if x0 - w > 0 and y0 - w > 0:
                        x = np.arange(x0 - w, x0 + w + 1)
                        y = np.arange(y0 - w, y0 + w + 1)
                        xgrid, ygrid = np.meshgrid(x, y)

                        cropped_image = img[y0 - w:y0 + w + 1, x0 - w:x0 + w + 1]

                        thetitle = title + " : Validated circle  id={} :: fitted (x0,y0) = ({},{}) , ".format(idx0, x0,
                                                                                                              y0)

                        fig = plt.figure(figsize=figsize)
                        ax = fig.add_subplot(111, projection='3d')
                        ax.view_init(45, -45)
                        ax.plot_surface(xgrid, ygrid, cropped_image, cmap=cmap)
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        plt.suptitle(thetitle)

                        plt.show()

            index += 1
        # end loop

    #-------------------------------------------------------------------------------------------------------
    def get_optimum_center(self,img, title="lambda_plus",figsize=[8, 8],cmap="terrain"):
        """

        :param img:
        :return:
        """

        self.my_logger.info(f'\n\t optimize centering')


        w = int(parameters.VIGNETTE_SIZE / 2)

        index = 0
        # loop on circles
        for circle in self.circles:
            # if the validation of circles has proceed
            if len(self.flag_validated_circles) > 0:
                if self.flag_validated_circles[index]:
                    y0 = int(circle.y_fit)
                    x0 = int(circle.x_fit)
                    idx0 = circle.index

                    #additionnal constraint on circle (x,y fit had to be done)
                    if x0-w > 0 and y0-w > 0 :

                        x = np.arange(x0-w, x0+w+1)
                        y = np.arange(y0-w, y0+w+1)
                        xgrid, ygrid = np.meshgrid(x, y)

                        cropped_image=img[y0-w:y0+w+1,x0-w:x0+w+1]


                        thetitle = title + " : Validated circle  id={} :: fitted (x0,y0) = ({},{}) , ".format(idx0, x0, y0)

                        fig = plt.figure(figsize=figsize)
                        ax = fig.add_subplot(111, projection='3d')
                        ax.view_init(45, -45)
                        ax.plot_surface(xgrid, ygrid, cropped_image, cmap=cmap)
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        plt.suptitle(thetitle)

                        plt.show()





            index+=1
        # end loop