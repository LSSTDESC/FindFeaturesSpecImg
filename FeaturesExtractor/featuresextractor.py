from FeaturesExtractor.config import *

from FeaturesExtractor.features.images import *
from FeaturesExtractor.features.features import *
from FeaturesExtractor.tools import *
from FeaturesExtractor import parameters

import sys

#---------------------------------------------------------------------------------------------------------------------
def FeatureExtractor(file_name, output_directory, config='./config/picdumidi.ini'):
    """ Spectractor
    Main function to extract a spectrum from an image

    Parameters
    ----------
    file_name: str
        Input file nam of the image to analyse
    output_directory: str
        Output directory

    Returns
    -------


    Examples
    --------

    """


    #--------- Start Logger
    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart FeatureExtractor')

    #--------- Load config file ---------------
    load_config(config)

    #-------- Load reduced image
    image = Image(file_name)

    # Set output path
    ensure_dir(output_directory)


    if parameters.DEBUG and parameters.FLAG_PLOT_IMG :
        image.plot_image(scale='log',title="Original image",cmap=plt.cm.gray)

    #-----------------------------------------------------------------------------------------------------------
    # Process the whole image by calculating Hessian, and its Eigen values images (lambda_plus and lambda_minus) and theta as well
    #-----------------------------------------------------------------------------------------------------------
    image.process_image()

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_PLUS:
        image.plot_image(img_type="lambda_p",scale='log',title="lambda_plus",cmap=plt.cm.gray)

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_MINUS:
        image.plot_image(img_type="lambda_m", scale='log', title="lambda_minus",cmap=plt.cm.gray)

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_THETA:
        image.plot_image(img_type="theta", scale='lin', title="theta",cmap=plt.cm.gray)

    # ----------------------------------------------------------------------------------------------------------
    # Clip Minimal and maximal values inside the images
    #------------------------------------------------------------------------------------------------------------
    image.clip_images()

    if parameters.DEBUG and parameters.FLAG_PLOT_IMG_CLIP:
        image.plot_image(img_type="img_cut",scale='log',title="Clipped image cut",cmap=plt.cm.gray)

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_PLUS_CLIP:
        image.plot_image(img_type="lambda_p_cut",scale='log',title="Clipped lambda_plus",cmap=plt.cm.gray)

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_MINUS_CLIP:
        image.plot_image(img_type="lambda_m_cut", scale='log', title="Clipped lambda_minus",cmap=plt.cm.gray)

    if parameters.DEBUG and parameters.FLAG_PLOT_THETA_CLIP:
        image.plot_image(img_type="theta_cut", scale='lin', title="Clipped theta",cmap=plt.cm.gray)

    #-------------------------------------------------------------------------------------------------------------
    # Calculate edges inside the images on lambda_plus and lambda_minus
    #-------------------------------------------------------------------------------------------------------------
    image.compute_edges()

    if parameters.DEBUG and ( parameters.FLAG_PLOT_LAMBDA_MINUS_EDGES or parameters.FLAG_PLOT_LAMBDA_PLUS_EDGES):
        image.plot_edges()


    #-------------------------------------------------------------------------------------------------------------
    # Detect the main features on the images.
    # Features are linear tracks and circles
    #-------------------------------------------------------------------------------------------------------------

    # work on lambda_plus
    # ---------------------
    image_features_lambdaplus=FeatureImage(image.img_cube[parameters.IndexImg.lambda_plus_edges])

    # detect lines in lambda_plus
    #----------------------------
    image_features_lambdaplus.find_lines()
    image_features_lambdaplus.plot_lines(image.img_cube[parameters.IndexImg.lambda_plus],scale="log",cmap=plt.cm.gray,
                                         title="Hough Lines detected in lambda_plus edges")


    # erase lines in edge image skeleton (canny) before search of circles
    # ---------------------------------------------
    image_features_lambdaplus.erase_lines()



    # detect circles in edge image cleaned from linear tracks
    #--------------------------------------------------------
    image_features_lambdaplus.find_circles()
    #image_features_lambdaplus.plot_circles(image.img_cube[parameters.IndexImg.img], scale="log",cmap=plt.cm.gray,
    #                                     title="Hough circles detected in lambda_plus edges")


    # calculate signal sum in circles to identify circiles associated with true sources
    # ----------------------------------------------------------------------------------
    thesignalsum=image_features_lambdaplus.compute_signal_in_circles(image.img_cube[parameters.IndexImg.img])



    # circle crossing : compute how many times extrapolated lines from identified linear tracks intersect circles and howmany pixels are in circles
    # --------------------------------------------------------------------------------------------------------------------------------------------
    NumberOfCrossings, NumberOfPixels = image_features_lambdaplus.compute_line_in_circles(image.img_cube[parameters.IndexImg.lambda_plus])



    # Validate circles : select the circles beeing crossed by enough extrapolated lines and having enough signal in it
    # -----------------------------------------------------------------------------------------------------------------------------
    image_features_lambdaplus.flag_validate_circles()

    # Validate line segments : only segment which extrapolation intersect validated circles are validated
    #----------------------------------------------------------------------------------------------------
    image_features_lambdaplus.flag_validate_lines()

    # show which circles are validated
    # ----------------------------------
    image_features_lambdaplus.plot_circles(image.img_cube[parameters.IndexImg.lambda_plus], scale="log", cmap=plt.cm.gray,
                                           title="Hough circles detected-validated in lambda_plus edges")



    # check validated line segments
    #-------------------------------
    image_features_lambdaplus.plot_validated_lines(img=image.img_cube[parameters.IndexImg.lambda_plus])

    # calculate the average angle of the segment tracks associated to each validated segment
    #----------------------------------------------------------------------------------------
    image_features_lambdaplus.compute_theta()


    # circle profile : look at each validated circles and search for a better circle center from light profile in X and Y
    # a fitted center position on X-profile and Y profile is performed
    #-------------------------------------------------------------------------------------------------------------------
    image_features_lambdaplus.get_circles_inprofiles(image.img_cube[parameters.IndexImg.lambda_plus])


    # Test if some validated circle have some saturation
    #---------------------------------------------------------
    image_features_lambdaplus.test_incircle_saturation(image.img_cube[parameters.IndexImg.img],title="original image",cmap="jet")


    # optimization for minimum
    # --------------------------
    image_features_lambdaplus.get_optimum_center(image.img_cube[parameters.IndexImg.lambda_plus],title="lambda_plus",cmap="jet")
    image_features_lambdaplus.get_optimum_center(image.img_cube[parameters.IndexImg.lambda_minus],title="lambda_minus",cmap="jet",optimize_flag=True)




    # write surmmary table on circles
    #-----------------------------------------
    image_features_lambdaplus.circlesummary.write(os.path.join(output_directory,parameters.FILENAME_LP_SUMMARYTABLE),
                                                  format='ascii',overwrite=True)







    # ---------------------------- lambda_minus
    # work on lambda_minus
    #image_features_lambdaminus = FeatureImage(image.img_cube[parameters.IndexImg.lambda_minus_edges])

    # detect lines in lambda_plus
    #image_features_lambdaminus.find_lines()
    #image_features_lambdaplus.plot_lines(image.img_cube[parameters.IndexImg.lambda_minus], scale="log", cmap=plt.cm.gray,
    #                                     title="Hough Lines detected in lambda_minus edges")

    # detect circles
    #image_features_lambdaminus.find_circles()
    #image_features_lambdaminus.plot_circles(image.img_cube[parameters.IndexImg.img], scale="log", cmap=plt.cm.gray,
    #                                       title="Hough circles detected in lambda_minus edges")

    # signal sum in circles
    #thesignalsum = image_features_lambdaminus.compute_signal_in_circles(image.img_cube[parameters.IndexImg.img])

    #print("LAMBDA_MINUS::SIGNALSUM = ", thesignalsum)

    # circle crossing
    #NumberOfCrossings, NumberOfPixels = image_features_lambdaminus.compute_line_in_circles()

    #print("LAMBDA_MINUS::NUMBEROFCROSSING = ", NumberOfCrossings)
    #print("LAMBDA_MINUS::NUMBEROFPIXEL    = ", NumberOfPixels)

    # Validate circles to be used
    #image_features_lambdaminus.flag_validate_circles()
    # Validate line segments
    #image_features_lambdaminus.flag_validate_lines()

    # plot
    #image_features_lambdaminus.plot_circles(image.img_cube[parameters.IndexImg.lambda_minus], scale="log", cmap=plt.cm.gray,
    #                                      title="Hough circles detected-validated in lambda_minus edges")

    # circle profile
    #image_features_lambdaplus.plot_circles_profiles(image.img_cube[parameters.IndexImg.lambda_minus])





