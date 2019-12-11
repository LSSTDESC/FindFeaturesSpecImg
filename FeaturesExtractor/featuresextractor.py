from FeaturesExtractor.config import *

from FeaturesExtractor.features.images import *
from FeaturesExtractor.features.features import *
from FeaturesExtractor.tools import *
from FeaturesExtractor import parameters




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

    #--------- Load config file
    load_config(config)

    #-------- Load reduced image
    image = Image(file_name)



    if parameters.DEBUG and parameters.FLAG_PLOT_IMG :
        image.plot_image(scale='log',title="Original image")

    #-------
    image.process_image()

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_PLUS:
        image.plot_image(img_type="lambda_p",scale='log',title="lambda_plus")

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_MINUS:
        image.plot_image(img_type="lambda_m", scale='log', title="lambda_minus")

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_THETA:
        image.plot_image(img_type="theta", scale='lin', title="theta")

    # -------
    image.clip_images()

    if parameters.DEBUG and parameters.FLAG_PLOT_IMG_CLIP:
        image.plot_image(img_type="img_cut",scale='log',title="Clipped image cut")

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_PLUS_CLIP:
        image.plot_image(img_type="lambda_p_cut",scale='log',title="Clipped lambda_plus")

    if parameters.DEBUG and parameters.FLAG_PLOT_LAMBDA_MINUS_CLIP:
        image.plot_image(img_type="lambda_m_cut", scale='log', title="Clipped lambda_minus")

    if parameters.DEBUG and parameters.FLAG_PLOT_THETA_CLIP:
        image.plot_image(img_type="theta_cut", scale='lin', title="Clipped theta")

    #-------------------------
    image.compute_edges()

    if parameters.DEBUG and ( parameters.FLAG_PLOT_LAMBDA_MINUS_EDGES or parameters.FLAG_PLOT_LAMBDA_PLUS_EDGES):
        image.plot_edges()


    #----------------------------
    # work on lambda_plus
    image_features_lambdaplus=FeatureImage(image.img_cube[parameters.IndexImg.lambda_plus_edges])

    # detect lines in lambda_plus
    image_features_lambdaplus.find_lines()
    image_features_lambdaplus.plot_lines(image.img_cube[parameters.IndexImg.img],scale="log",cmap=plt.cm.gray,
                                         title="Hough Lines detected in lambda_plus edges")

    # detect circles
    image_features_lambdaplus.find_circles()
    image_features_lambdaplus.plot_circles(image.img_cube[parameters.IndexImg.img], scale="log",cmap=plt.cm.gray,
                                         title="Hough circles detected in lambda_plus edges")

    thesignalsum=image_features_lambdaplus.compute_signal_in_circles(image.img_cube[parameters.IndexImg.img])

    print("SIGNALSUM=",thesignalsum)


    # ----------------------------
    # work on lambda_minus
    image_features_lambdaminus = FeatureImage(image.img_cube[parameters.IndexImg.lambda_minus_edges])

    # detect lines in lambda_plus
    image_features_lambdaminus.find_lines()
    image_features_lambdaplus.plot_lines(image.img_cube[parameters.IndexImg.img], scale="log", cmap=plt.cm.gray,
                                         title="Hough Lines detected in lambda_minus edges")

    # detect circles
    image_features_lambdaminus.find_circles()
    image_features_lambdaminus.plot_circles(image.img_cube[parameters.IndexImg.img], scale="log", cmap=plt.cm.gray,
                                           title="Hough circles detected in lambda_minus edges")






    # Set output path
    ensure_dir(output_directory)


