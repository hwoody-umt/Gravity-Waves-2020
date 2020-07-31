from WaveDetectionFunctions import getAllUserInput, cleanData, readFromData, interpolateData, findPeakSquare, \
    waveletTransform, findPeaks, displayProgress, findPeakRegion, removePeaks, updatePlotter, invertWaveletTransform, \
    getParameters, drawPowerSurface
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



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
    spatialResolution = 5  # meters in between uniformly distributed data points, must be pos integer
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
    # Check to see if this is done by Zink & Vincent or Murphy et al
    peakRemovalMask = np.zeros(wavelets.get('power').shape, dtype=bool)
    for peak in peaks:
        if wavelets.get('wavelengths')[peak[0]] > wavelets.get('coi')[peak[1]]:
            peakRemovalMask[peak[0], peak[1]] = True
    peaks = removePeaks(peakRemovalMask, peaks)

    numPeaks = len(peaks)  # To keep track of progress
    peaksToPlot = peaks.copy()  # Keep peaks for plot at end
    colorsToPlot = np.array(['blue'] * len(peaks))  # Keep track for plots at end



    yScale = wavelets.get('wavelengths')
    # plt.imshow(wavelets.get('power'), extent=extents, origin='lower')
    plt.contourf(data['Alt'] / 1000, yScale, wavelets.get('power'), levels=50)
    cb = plt.colorbar()
    plt.scatter(data['Alt'][peaksToPlot.T[1]] / 1000, yScale[peaksToPlot.T[0]], c='red', marker='.')
    plt.plot(data['Alt']/1000, wavelets.get('coi'), color='blue')
    # plt.contour(plotter, colors='red', levels=[0.5])
    # ax.set_aspect('auto')
    plt.yscale("log")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Vertical Wavelength [m]")
    plt.ylim(yScale[0], yScale[-1])
    plt.title("Power Surface with Local Maxima")
    cb.set_label("Power [m^2/s^2]")
    plt.savefig("C:/Users/12069/Documents/Eclipse2020/Presentation/Python Images/Power_Surface_Example.png")
    plt.close()

    # Numpy array for plotting purposes
    plotter = np.zeros( wavelets.get('power').shape, dtype=bool )

    # Index to only save 1/50 of the data for plotting, the detail isn't all needed
    trimIndex = np.arange(0, len(data['Time']), 50)
    waves = {
        'waves': {},  # Empty dictionary, to contain wave characteristics
        'flightPath': {  # Flight path for plotting results
            'time': np.array(data['Time'][trimIndex]).tolist(),
            'alt': np.array(data['Alt'][trimIndex]).tolist()
        }
    }
    waveCount = 1  # For naming output waves

    # Iterate over local maximums to identify wave characteristics
    while len(peaks) > 0:
        # Output progress to console, keeping user in the loop
        displayProgress( peaks, numPeaks )
        # Identify the region surrounding the peak
        region = findPeakSquare( wavelets.get('power'), peaks[0])

        # Save for plotting
        currentPeak = peaks[0]

        # Update list of peaks that have yet to be analyzed
        peaks = removePeaks( region, peaks )

        if region.sum().sum() <= 1:  # Only found the peak, not the region
            continue  # Don't bother analyzing the single cell

        #ax = plt.axes()
        #plt.imshow(wavelets.get('power'))  # ,
        # extent=extents)
        #ax.set_aspect('auto')
        #cb = plt.colorbar()
        # plt.contour(data['Alt'] / 1000, np.flip(yScale), plotter,
        #            colors='red')
        # plt.scatter(data['Alt'][peaksToPlot.T[1]] / 1000, np.flip(yScale)[peaksToPlot.T[0]], marker='.',
        #            edgecolors='red')
        #plt.scatter(peaksToPlot[:, 1], peaksToPlot[:, 0], c=colorsToPlot, marker='*')
        #plt.contour(region, colors='red', levels=[0.5])
        #plt.xlabel("Altitude [index]")
        #plt.ylabel("Vertical Wavelength [index]")
        #plt.title("Power surface, including traced peaks")
        #cb.set_label("Power [m^2/s^2]")
        #plt.show()

        # Update plotting mask
        plotter = updatePlotter( region, plotter )

        # Get inverted regional maximums
        wave = invertWaveletTransform( region, wavelets )

        # Perform analysis to find wave information
        parameters = getParameters( data, wave, spatialResolution, region, currentPeak[1], wavelets.get('wavelengths')[currentPeak[0]], file[0:5] )
        # If found, save parameters to dictionary of waves
        if parameters:
            name = 'wave' + str(waveCount)
            waves['waves'][name] = parameters
            waveCount += 1
            colorIndex = (currentPeak == peaksToPlot).sum(axis=1)
            colorsToPlot[np.where(colorIndex == 2)] = 'red'

    # Save waves data here, if saveData boolean is true
    if userInput.get('saveData'):
        # Sort the data first, for easier analysis
        waves['waves'] = sorted(waves['waves'].items(), key=lambda x: x[1].get('Altitude [km]'))
        # Save waves data here, if saveData boolean is true
        with open(userInput.get('savePath') + "/" + file[0:-4] + '_wave_parameters.json', 'w') as writeFile:
            json.dump(waves, writeFile, indent=4, default=str)
    else:
        print("\nWave parameters found:")
        print(waves['waves'])

    # Also, build nice output plot
    drawPowerSurface(userInput, file, wavelets, data['Alt']/1000, plotter, peaksToPlot, colorsToPlot)

    print("\nFinished file analysis")

########## FINISHED ANALYSIS ##########
print("\nAnalyzed all files in folder "+userInput.get('dataSource')+"/")