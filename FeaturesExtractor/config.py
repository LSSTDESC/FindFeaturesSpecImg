import configparser
import os
import sys
import re
import numpy as np
import coloredlogs
import logging
import astropy.units as units

import parameters

logging.getLogger("matplotlib").setLevel(logging.ERROR)



def load_config(config_filename):
    if not os.path.isfile(config_filename):
        sys.exit('Config file %s does not exist.' % config_filename)
    # Load the configuration file
    config = configparser.ConfigParser()
    config.read(config_filename)

    # List all contents
    for section in config.sections():
        for options in config.options(section):
            value = config.get(section, options)
            if re.match("[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", value):
                if ' ' in value:
                    value = str(value)
                elif '.' in value or 'e' in value:
                    value = float(value)
                else:
                    value = int(value)
            elif value == 'True' or value == 'False':
                value = config.getboolean(section, options)
            else:
                value = str(value)
            setattr(parameters, options.upper(), value)

    # Derive other parameters
    parameters.MY_FORMAT = "%(asctime)-20s %(name)-10s %(funcName)-20s %(levelname)-6s %(message)s"
    logging.basicConfig(format=parameters.MY_FORMAT, level=logging.WARNING)
    mypath = os.path.dirname(__file__)
    # may overwrite parameters

    if parameters.VERBOSE:
        for section in config.sections():
            print("Section: %s" % section)
            for options in config.options(section):
                value = config.get(section, options)
                par = getattr(parameters, options.upper())
                print(f"x {options}: {value}\t => parameters.{options.upper()}: {par}\t {type(par)}")




def set_logger(logger):
    """Logger function for all classes.

    Parameters
    ----------
    logger: str
        Name of the class, usually self.__class__.__name__

    Returns
    -------
    my_logger: logging
        Logging object

    Examples
    --------
    >>> class Test:
    ...     def __init__(self):
    ...         self.my_logger = set_logger(self.__class__.__name__)
    ...     def log(self):
    ...         self.my_logger.info('This info test function works.')
    ...         self.my_logger.debug('This debug test function works.')
    ...         self.my_logger.warning('This warning test function works.')
    >>> test = Test()
    >>> test.log()
    """
    my_logger = logging.getLogger(logger)
    if parameters.VERBOSE > 0:
        my_logger.setLevel(logging.INFO)
        coloredlogs.install(fmt=parameters.MY_FORMAT, level=logging.INFO)
    else:
        my_logger.setLevel(logging.WARNING)
        coloredlogs.install(fmt=parameters.MY_FORMAT, level=logging.WARNING)
    if parameters.DEBUG:
        my_logger.setLevel(logging.DEBUG)
        coloredlogs.install(fmt=parameters.MY_FORMAT, level=logging.DEBUG)
    return my_logger

