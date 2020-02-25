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






    # ---------------------------- lambda_minus
    # work on lambda_minus
    # ---------------------



    image_features_lambdaminus = FeatureImage(image.img_cube[parameters.IndexImg.lambda_minus_edges])

    # detect lines in lambda_plus
    # ----------------------------
    image_features_lambdaminus.find_lines()
    image_features_lambdaminus.plot_lines(image.img_cube[parameters.IndexImg.lambda_minus], scale="log", cmap=plt.cm.gray,
                                         title="Hough Lines detected in lambda_minus edges")








    # copy circles in lambda_plus into lambda_minus (bad search of circles in lambda_minus)
    #--------------------------------------------------------------------------------------
    circles,signal,numberoflines,numberofpoints,flag_validated_circles,flag_saturation_circles,circlesummary = image_features_lambdaplus.get_circles()

    image_features_lambdaminus.set_circles(circles,signal,numberoflines,numberofpoints,flag_validated_circles,flag_saturation_circles,circlesummary)

    # Show circles from lambda_plus into lambda_minus
    # --------------------------------------------------------
    image_features_lambdaminus.plot_circles(image.img_cube[parameters.IndexImg.img], scale="log",cmap=plt.cm.gray,
                                         title="Hough circles shown in lambda_minus (found in lambda_plus edges)")



    # Validate line segments : only segment which extrapolation intersect validated circles are validated
    # ----------------------------------------------------------------------------------------------------
    image_features_lambdaminus.flag_validate_lines()


    # check validated line segments
    # -------------------------------
    image_features_lambdaminus.plot_validated_lines(img=image.img_cube[parameters.IndexImg.lambda_minus])


    # Search for aigrettes
    #---------------------------
    flag_aigrettes_found  = image_features_lambdaminus.flag_validate_aigrettelines()




    if flag_aigrettes_found:
        image_features_lambdaminus.plot_aigrettevalidated_lines(img=image.img_cube[parameters.IndexImg.lambda_minus])
        image_features_lambdaminus.compute_aigrettes_center(img=image.img_cube[parameters.IndexImg.lambda_plus],title= "fit aigret, showing lambda_plus",cmap=plt.cm.gray)
        image_features_lambdaminus.compute_aigrettes_center(img=image.img_cube[parameters.IndexImg.lambda_minus],title="fit aigret, showing lambda_minus",cmap=plt.cm.gray)






