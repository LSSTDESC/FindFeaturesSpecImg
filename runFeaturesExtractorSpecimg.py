
from FeaturesExtractor import parameters
from FeaturesExtractor import featuresextractor


import os
import sys
import pandas as pd

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(dest="input", metavar='path', default=["T1M_20190215_225550_730_HD116405_Filtre_None_bin1x1.1_red.fit"],
                        help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                        nargs='*')
    parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                        help="Enter debug mode (more verbose and plots).", default=False)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Enter verbose (print more stuff).", default=False)
    parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                        help="Write results in given output directory (default: ./outputs/).")
    parser.add_argument("-c", "--config", dest="config", default="config/picdumidi.ini",
                        help="INI config file. (default: config.picdumidi.ini).")
    args = parser.parse_args()

    parameters.VERBOSE = args.verbose

    if args.debug:
        parameters.DEBUG = True
        parameters.VERBOSE = True

    file_names = args.input
    config_name = args.config



