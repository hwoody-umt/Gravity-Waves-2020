# Import all functions from 'WaveDetectionFunctions.py', file which holds functions for this program
from WaveDetectionFunctions import *

# Import os to iterate through files in given directory
import os


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

    # Get the launch time and pbl height from profile header
    launchDateTime, pblHeight = readFromData( file, userInput.get('dataSource'))
    # Set the height in between data points, currently 5 because it's the nominal data acquisition rate
    spatialResolution = 5  # meters, must be pos integer
    # Interpolate to create a uniform spatial grid of data, rather than temporal
    data = interpolateData( data, spatialResolution, pblHeight, launchDateTime )

    # If nothing was returned, file was missing too much data,
    # so skip ahead to the next loop iteration
    if data.empty:
        continue

    # Perform the continuous wavelet transform to get the power surface
    wavelets = waveletTransform( data, spatialResolution, 'MORLET')  # Use morlet wavelet

    # Find local maxima in power surface
    peaks = findPeaks( wavelets.get('power') )

    # Filter local maxima to within cone of influence, per Torrence & Compo (1998)
    # I need to check to see if this is done by Zink & Vincent or Murphy et al
    peaks = filterPeaksCOI(wavelets, peaks)

    # Define the needed variables, outside of the loop
    waves, plottingInfo = setUpLoop(data, wavelets, peaks)


    # Iterate over local maxima to identify wave characteristics
    while len(peaks) > 0:

        # Output progress to console, keeping user informed
        displayProgress( peaks, len(plottingInfo.get('peaks')) )

        # Identify the region surrounding the peak
        region = findPeakSquare( wavelets.get('power'), peaks[0])

        #if region.sum().sum() <= 1:  # Only found the peak, not the region
        #    continue  # Don't bother analyzing the single cell

        # Get inverted regional maximums
        wave = invertWaveletTransform( region, wavelets )

        # Perform analysis to find wave information
        parameters = getParameters(data, wave, spatialResolution, peaks[0, 1], wavelets.get('wavelengths')[peaks[0, 0]])

        # Update pertinent variables to save current wave parameters, increment counters, and shorten peaks list
        waves, plottingInfo, peaks = saveParametersInLoop(waves, plottingInfo, parameters, region, peaks)


    # Either save or print to the console wave parameters, depending on user input
    outputWaveParameters(userInput, waves, file)

    # Also, build and show/save power surface plot
    drawPowerSurface(userInput, file, wavelets, data['Alt']/1000, plottingInfo.get('regions'), plottingInfo.get('peaks'), plottingInfo.get('colors'))

    # Done with this radiosonde flight, print 'finished' and continue to next file
    print("\nFinished file analysis")

########## FINISHED ANALYSIS ##########
print("\nAnalyzed all files in folder "+userInput.get('dataSource')+"/")
