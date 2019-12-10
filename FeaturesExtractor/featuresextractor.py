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
    my_logger.info('\n\tStart SPECTRACTOR')
    # Load config file
    load_config(config)

    # Load reduced image
    image = Image(file_name, target=target, disperser_label=disperser_label,logbook=logbook)

    if parameters.DEBUG:
        image.plot_image(scale='log10', target_pixcoords=guess)

    # Set output path
    ensure_dir(output_directory)

    #test filename output : fit or fits

    filetype=file_name.split('.')[-1]

    if filetype=="fits":
        #output_filename = file_name.split('/')[-1]
        output_filename=os.path.basename(file_name)
        output_filename = output_filename.replace('.fits', '_spectrum.fits')
        output_filename = output_filename.replace('.fz', '_spectrum.fits')
        output_filename = os.path.join(output_directory, output_filename)
        output_filename_spectrogram = output_filename.replace('spectrum','spectrogram')
        output_filename_psf = output_filename.replace('spectrum.fits','table.csv')
    elif filetype=="fit":
        output_filename = os.path.basename(file_name)
        output_filename = output_filename.replace('.fit', '_spectrum.fits')
        output_filename = output_filename.replace('.fz', '_spectrum.fits')
        output_filename = os.path.join(output_directory, output_filename)
        output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
        output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')
    else:
        output_filename = os.path.basename(file_name)
        output_filename = output_filename.replace('.fits', '_spectrum.fits')
        output_filename = output_filename.replace('.fz', '_spectrum.fits')
        output_filename = os.path.join(output_directory, output_filename)
        output_filename_spectrogram = output_filename.replace('spectrum', 'spectrogram')
        output_filename_psf = output_filename.replace('spectrum.fits', 'table.csv')


    # Find the exact target position in the raw cut image: several methods
    my_logger.info('\n\tSearch for the target in the original image...')
    target_pixcoords = find_target(image, guess)

    # Rotate the image: several methods
    turn_image(image)

    # Find the exact target position in the rotated image: several methods
    my_logger.info('\n\tSearch for the target in the rotated image...')

    target_pixcoords_rotated = find_target(image, guess, rotated=True)

    # Create Spectrum object
    spectrum = Spectrum(image=image)

    # Subtract background and bad pixels
    if parameters.OBS_NAME != 'PICDUMIDI':
        extract_spectrum_from_image(image, spectrum, w=parameters.PIXWIDTH_SIGNAL,
                                ws = (parameters.PIXDIST_BACKGROUND,
                                      parameters.PIXDIST_BACKGROUND+parameters.PIXWIDTH_BACKGROUND),
                                right_edge=parameters.CCD_IMSIZE-200)
    else:
        alpha_rot=image.rotation_angle*np.pi/180.
        right_edge_max=int(parameters.CCD_IMSIZE/np.cos(alpha_rot))-100
        extract_spectrum_from_image(image, spectrum, w=parameters.PIXWIDTH_SIGNAL,
                                    ws=(parameters.PIXDIST_BACKGROUND,
                                        parameters.PIXDIST_BACKGROUND + parameters.PIXWIDTH_BACKGROUND),
                                    right_edge=right_edge_max)



    spectrum.atmospheric_lines = atmospheric_lines
    # Calibrate the spectrum
    calibrate_spectrum(spectrum)
    if line_detection:
        my_logger.info('\n\tCalibrating order %d spectrum...' % spectrum.order)
        calibrate_spectrum_with_lines(spectrum)
    else:
        spectrum.header['WARNINGS'] = 'No calibration procedure with spectral features.'

    # Save the spectrum
    my_logger.info('\n\tSave Spectrum in {output_filename}...')
    my_logger.info('\n\tSave Spectrogram in {output_filename_spectrogram}...')

    spectrum.save_spectrum(output_filename, overwrite=True)
    spectrum.save_spectrogram(output_filename_spectrogram, overwrite=True)

    # Plot the spectrum
    if parameters.VERBOSE and parameters.DISPLAY:
        spectrum.plot_spectrum(xlim=None)
    distance = spectrum.chromatic_psf.get_distance_along_dispersion_axis()
    spectrum.lambdas = np.interp(distance, spectrum.pixels, spectrum.lambdas)
    spectrum.chromatic_psf.table['lambdas'] = spectrum.lambdas
    spectrum.chromatic_psf.table.write(output_filename_psf, overwrite=True)
    return spectrum