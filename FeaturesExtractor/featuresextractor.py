from FeaturesExtractor.config import *

from FeaturesExtractor.features.images import *
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

    my_logger = set_logger(__name__)
    my_logger.info('\n\tStart FeatureExtractor')

    # Load config file
    load_config(config)

    # Load reduced image
    image = Image(file_name)



    if parameters.DEBUG:
        image.plot_image(scale='log',title="Original image")

    image.process_image()

    if parameters.DEBUG:
        image.plot_image(img_type="lambda_p",scale='log',title="lambda_plus")

    if parameters.DEBUG:
        image.plot_image(img_type="lambda_m", scale='log', title="lambda_minus")

    # Set output path
    ensure_dir(output_directory)


