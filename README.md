# FindFeaturesSpecImg

The goal of **FindFeaturesSpecImg** is to identify main geometrical features in
a Spectral Image.

Its main application is the auxiliary telescope of LSST aiming at measuring star spectra
to extract atmospheric transmission.

In particular it is proposed to feed **Spectractor** with the diffraction order 0 Star position in the image included
the orientation angle of its first 

- author : Sylvie Dagoret-Campagne
- affiliation : LAL/IN2P3/CNRS puis IJCLAB/IN2P3/CNRS
- creation date : December 10 th 2019 
- update : December 13 th 2019 
- update : February 25th 2020



## Installation


Spectractor is written in Python 3.7 It needs the numpy, skimage, astropy, modules for science computations, and also logging and coloredlogs. 



### External dependencies

- numpy
- matplotlib
- astropy
- skimage

...

## Basic usage

in a shell, run the command :

### For an Image taken at a telescope
python runFeaturesExtractorSpecimg.py input_image_filename

or

### For an Image taken from a lamp at optical testbench 
python runFeaturesExtractorSpecLampimg.py  -c config/opticalbench.ini tests/data/10_CCD1_20200206164429_red.fits 


- **input**  : fit filename of the input image
- **output** : an ascii file containing the position of the identified order 0 stars, plus some criteria
 Notice some default parameters are in file parameters.py, which may be changed in config files
 
 - more input arguments :
 
   - config file path 
   - 
   
## What it does ?
   
### Launcher 1 : start with lambda_plus

- runFeaturesExtractorSpecimg.py : Original Image analyser

#### It sequence is programmed in :

- FeaturesExtractor/featuresextractor.py	

### Launcher 2 : start with lambda_minus

- runFeaturesExtractorSpecLampimg.py : Analyser developped for test bench
   
#### Its sequence is programmed in:

- FeaturesExtractor/featuresextractor_lambdam.py
   
## Software organisation


