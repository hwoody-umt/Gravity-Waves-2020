from WaveDetectionFunctions import getAllUserInput, cleanData, readFromData, interpolateData, drawPlots, waveletTransform, findPeaks, displayProgress, findPeakRegion, removePeaks, updatePlotter, invertWaveletTransform, getParameters
import numpy as np
import os
import json
import matplotlib.pyplot as plt


########## ACTUAL RUNNING CODE ##########

# First, get applicable user input.
userInput = getAllUserInput()

# Then, iterate over files in data directory
for file in os.listdir( userInput.get('dataSource') ):
    # Import and clean the data, given the file path
    data = cleanData( file, userInput.get('dataSource') )

    # If nothing was returned, file is not recognized as a GRAWMET profile,
    # so skip ahead to the next loop iteration
    if data.empty:
        continue

    launchDateTime, pblHeight = readFromData( file, userInput.get('dataSource'))
    spatialResolution = 4  # meters in between uniformly distributed data points, must be pos integer
    data = interpolateData( data, spatialResolution, pblHeight, launchDateTime )

    # If nothing was returned, file was missing too much data,
    # so skip ahead to the next loop iteration
    if data.empty:
        continue

    # Get the stuff... comment better later!
    wavelets = waveletTransform( data, spatialResolution, 'MORLET')  # Use morlet wavelet

    # For current debugging, remove when plotting is fixed
    #print("Wavelengths run from " + str(wavelets.get('wavelengths')[0]/1000) + " to " + str(wavelets.get('wavelengths')[-1]/1000) + " kilometers.")

    # Find local maximums in power surface
    peaks = findPeaks( wavelets.get('power') )
    numPeaks = len(peaks)  # To keep track of progress
    peaksToPlot = peaks.copy()  # Keep peaks for plot at end
    colorsToPlot = np.array(['blue'] * peaks.shape[0])  # Keep track for plots at end

    # Numpy array for plotting purposes
    plotter = np.zeros( wavelets.get('power').shape, dtype=bool )

    waves = {
        'waves': {},  # Empty dictionary, to contain wave characteristics
        'flightPath': {'time': np.array(data['Time']).tolist(), 'alt': np.array(data['Alt']).tolist()}  # Flight path for plotting results
    }
    waveCount = 1  # For naming output waves

    # Iterate over local maximums to identify wave characteristics
    while len(peaks) > 0:
        # Output progress to console and increment counter
        displayProgress( peaks, numPeaks )
        # Identify the region surrounding the peak
        region = findPeakRegion( wavelets.get('power'), peaks[0], plotter )

        # Update list of peaks that have yet to be analyzed
        peaks = removePeaks( region, peaks )

        if region.sum().sum() <= 1:  # Only found the peak, not the region
            continue  # Don't bother analyzing the single cell

        # Update plotting mask
        plotter = updatePlotter( region, plotter )

        # Get inverted regional maximums
        wave = invertWaveletTransform( region, wavelets )

        # Perform analysis to find wave information
        parameters = getParameters( data, wave, spatialResolution, region )
        # If found, save parameters to dictionary of waves
        if parameters:
            name = 'wave' + str(waveCount)
            waves['waves'][name] = parameters
            waveCount += 1
            colorIndex = (peaks[0] == peaksToPlot).sum(axis=1)
            colorsToPlot[np.where(colorIndex == 2)] = 'red'

    # Save waves data here, if saveData boolean is true
    if userInput.get('saveData'):
        # Save waves data here, if saveData boolean is true
        with open(userInput.get('savePath') + "/" + file[0:-4] + '_wave_parameters.json', 'w') as writeFile:
            json.dump(waves, writeFile, indent=4, default=str)

    # Also, build nice output plot

    #yScale = wavelets.get('wavelengths')

    #extents = [
    #    data['Alt'][0] / 1000,
    #    data['Alt'][len(data['Alt']) - 1] / 1000,
    #    yScale[0],
    #    yScale[len(yScale) - 1]
    #]
    ax = plt.axes()
    plt.imshow(wavelets.get('power'))#,
    #extent=extents)
    ax.set_aspect('auto')
    cb = plt.colorbar()
    #plt.contour(data['Alt'] / 1000, np.flip(yScale), plotter,
    #            colors='red')
    #plt.scatter(data['Alt'][peaksToPlot.T[1]] / 1000, np.flip(yScale)[peaksToPlot.T[0]], marker='.',
    #            edgecolors='red')
    plt.scatter(peaksToPlot[:, 1], peaksToPlot[:, 0], c=colorsToPlot, marker='*')
    plt.contour(plotter, colors='red', levels=[0.5])
    plt.xlabel("Altitude [index]")
    plt.ylabel("Vertical Wavelength [index]")
    plt.title("Power surface, including traced peaks")
    cb.set_label("Power [m^2/s^2]")

    if userInput.get('saveData'):
        plt.savefig(userInput.get('savePath') + "/" + file[0:-4] + "_power_surface.png")
    if userInput.get('showPlots'):
        plt.show()
    plt.close()
    print("\nFinished file analysis")

########## FINISHED ANALYSIS ##########
print("\nAnalyzed all files in folder "+userInput.get('dataSource')+"/")

