########## IMPORT LIBRARIES AND FUNCTIONS ##########

import numpy as np  # Numbers (like pi) and math
import matplotlib.pyplot as plt  # Easy plotting
import matplotlib.path as path  # Used for finding the peak region
import pandas as pd  # Convenient data formatting, and who doesn't want pandas
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import os  # File reading and input
from io import StringIO  # Used to run strings through input/output functions
from TorrenceCompoWavelets import wavelet as continuousWaveletTransform  # Torrence & Compo (1998) wavelet analysis code
from skimage.feature import peak_local_max  # Find local max
import datetime  # Turning time into dates
from skimage.measure import find_contours  # Find contour levels around local max
from scipy.ndimage.morphology import binary_fill_holes  # Then fill in those contour levels
from scipy.signal import argrelextrema



def drawPowerSurface(userInput, fileName, wavelets, altitudes, plotter, peaksToPlot, colorsToPlot):
    # FUNCTION PURPOSE: Create a power surface showing local maxima and their outlines
    #
    # INPUTS:
    #   userInput: Dictionary containing whether to save/show the plots, as well as a save path
    #   fileName: String, name of the profile file currently being analyzed
    #   wavelets: Dictionary containing power surface and corresponding wavelengths
    #   altitudes: Pandas DataFrame column with altitudes (IN KM) corresponding to the power surface
    #   plotter: Boolean mask identifying traced regions on power surface
    #   peaksToPlot: Numpy 2d array containing peaks, e.g. [ [x1, y1], [x2, y2], ... [xN, yN] ]
    #   colorsToPlot: Numpy array of strings corresponding to each peak, e.g. [ "color1", "color2", ... "colorN" ]
    #
    # OUTPUTS: Returns nothing, prints to console and saves files and/or shows images

    # If neither saving nor showing the plots, then don't bother making them
    if not userInput.get('saveData') and not userInput.get('showPlots'):
        return

    # Console output to keep the user up to date
    print("\r\nGenerating power surface plots", end='')

    # Get the vertical wavelengths for the Y coordinates
    yScale = wavelets.get('wavelengths')
    # Contourf is a filled contour, which is the easiest tool to plot a colored surface
    # Levels is set to 50 to make it nearly continuous, which takes a while,
    # but looks good and handles the non-uniform yScale, which plt.imshow() does not
    plt.contourf(altitudes, yScale, wavelets.get('power'), levels=50)
    # Create a colorbar for the z scale
    cb = plt.colorbar()
    # Plot the outlines of the local maxima, contour is an easy way to outline a mask
    # The 'plotter' is a boolean mask, so levels is set to 0.5 to be between 0 and 1
    plt.contour(altitudes, yScale, plotter, colors='red', levels=[0.5])
    # Make a scatter plot of the identified peaks, coloring them according to which ones were confirmed as waves
    plt.scatter(altitudes[peaksToPlot.T[1]], yScale[peaksToPlot.T[0]], c=colorsToPlot, marker='.')
    # Set the axis scales, labels, and titles
    plt.yscale("log")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Vertical Wavelength [m]")
    plt.title("Power surface, including traced peaks")
    cb.set_label("Power [m^2/s^2]")

    # Save and/or show the plot, according to user input.
    if userInput.get('saveData'):
        plt.savefig(userInput.get('savePath') + "/" + fileName[0:-4] + "_power_surface.png")
    if userInput.get('showPlots'):
        plt.show()
    plt.close()

    # Below is code to plot the power surface in 3D.
    # It's commented out because it doesn't look very good,
    # and it's confusing/not that useful.
    # However, with several innovations, it could be helpful,
    # so it's here for the future.

    #from matplotlib import cm
    #X, Y = np.meshgrid(altitudes, np.log10(yScale))
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #surf = ax.plot_surface(X, Y, wavelets.get('power'), cmap=cm.viridis)
    #fig.colorbar(surf)
    #ax.set_zlabel('Power [m^2(s^-2)]')
    #ax.set_ylabel('log10(vertical wavelength)')
    #ax.set_xlabel('Altitude [km]')
    #if userInput.get('saveData'):
    #    plt.savefig(userInput.get('savePath') + "/" + fileName[0:-4] + "_power_surface_3D.png")
    #if userInput.get('showPlots'):
    #    plt.show()
    #plt.close()


########## USER INPUT SECTION ##########


def getUserInputFile(prompt):
    # FUNCTION PURPOSE: Get a valid path (relative or absolute) to a directory from user
    #
    # INPUTS:
    #   prompt: String that is printed to the console to prompt user input
    #
    # OUTPUTS:
    #   userInput: String containing path to an existing directory

    # Print the prompt to console
    print(prompt)

    # userInput starts as empty string
    userInput = ""

    # While userInput remains empty, get input
    while not userInput:
        userInput = input()

        # If input isn't a valid directory, set userInput to empty string
        if not os.path.isdir(userInput):
            # Console output to let user know requirements
            print("Please enter a valid directory:")
            userInput = ""

    # Now that the loop has finished, userInput must be valid, so return
    return userInput


def getUserInputTF(prompt):
    # FUNCTION PURPOSE: Get a valid boolean (True or False) from user
    #
    # INPUTS:
    #   prompt: String that is printed to the console to prompt user input
    #
    # OUTPUTS:
    #   userInput: Boolean containing the user's answer to 'prompt'

    # Print the prompt to console, followed by the user's input options ("Y" or "N")
    print(prompt+" (Y/N)")

    # userInput starts as empty string
    userInput = ""

    # While userInput remains empty, get input
    while not userInput:
        userInput = input()
        # If input isn't either "Y" or "N", set userInput to empty string
        if lower(userInput) != "y" and lower(userInput) != "n":
            print("Please enter a valid answer (Y/N):")
            # Console output to let user know requirements
            userInput = ""

    # Now that the loop has finished, return True for "Y" and False for "N"
    if lower(userInput) == "y":
        return True
    else:
        return False


def getAllUserInput():
    # FUNCTION PURPOSE: Get all required user input to begin running the program
    #
    # INPUTS: None
    #
    # OUTPUTS:
    #   results: Dictionary containing the user answers to the 3 or 4 questions below

    # Get the directory containing the data for analysis
    dataSource = getUserInputFile("Enter path to data input directory: ")

    # Get a boolean value for whether to display the generated plots
    showPlots = getUserInputTF("Do you want to display plots for analysis?")

    # Get a boolean value for whether to save calculated data
    saveData = getUserInputTF("Do you want to save the output data?")

    # If saving the data, get the directory in which to save it
    if saveData:
        savePath = getUserInputFile("Enter path to data output directory: ")
    else:
        savePath = "NA"

    # Print results to inform user and begin program
    # Could eventually add a "verbose" option into user input that regulates print() commands
    print("Running with the following parameters:")
    print("Path to input data: "+dataSource+"/")
    print("Display plots: "+str(showPlots))
    print("Save data: "+str(saveData))
    if saveData:
        print("Path to output data: "+savePath+"/\n")
    else:
        # Extra line for improved readability
        print()

    # Build a dictionary to return values
    results = {
        'dataSource': dataSource,
        'showPlots': showPlots,
        'saveData': saveData
    }
    if saveData:
        results.update( {'savePath': savePath })

    # Return the resulting dictionary
    return results


########## DATA INPUT SECTION ##########

def cleanData(file, path):
    # FUNCTION PURPOSE: Read a data file, and if the file contains GRAWMET profile data,
    #                   then clean the data and return the results
    #
    # INPUTS:
    #   file: The filename of the data file to read
    #   path: The path (absolute or relative) to the file
    #
    # OUTPUTS:
    #   data: Pandas DataFrame containing the time [s], altitude [m], temperature [deg C],
    #           pressure [hPa], wind speed [m/s], wind direction [deg], latitude [decimal deg],
    #           and longitude [decimal deg] of the radiosonde flight


    # If file is not a txt file, end now
    if not file.endswith(".txt"):
        return pd.DataFrame()  # Empty data frame means end analysis


    # Open and investigate the file
    contents = ""
    isProfile = False  # Check to see if this is a GRAWMET profile
    f = open(os.path.join(path, file), 'r')
    print("\nOpening file "+file+":")
    for line in f:  # Iterate through file, line by line
        if line.rstrip() == "Profile Data:":
            isProfile = True  # We found the start of the real data in GRAWMET profile format
            contents = f.read()  # Read in rest of file, discarding header
            print("File contains GRAWMET profile data")
            break
    f.close()  # Need to close opened file

    # If we checked the whole file and didn't find it, end analysis now.
    if not isProfile:
        print("File "+file+" is either not a GRAWMET profile, or is corrupted.")
        return pd.DataFrame()

    # Read in the data and perform cleaning

    # Need to remove space so Virt. Temp reads as one column, not two
    contents = contents.replace("Virt. Temp", "Virt.Temp")
    # Break file apart into separate lines
    contents = contents.split("\n")
    contents.pop(1)  # Remove units so that we can read table
    index = -1  # Used to look for footer
    for i in range(0, len(contents)):  # Iterate through lines
        if contents[i].strip() == "Tropopauses:":
            index = i  # Record start of footer
    if index >= 0:  # Remove footer, if found
        contents = contents[:index]
    contents = "\n".join(contents)  # Reassemble string

    # Read in the data
    data = pd.read_csv(StringIO(contents), delim_whitespace=True)
    del contents  # Free up a little memory

    # Find the end of usable (ascent) data
    badRows = []  # Index, soon to contain any rows to be removed
    for row in range(data.shape[0]):  # Iterate through rows of data
        if not str(data['Rs'].loc[row]).replace('.', '', 1).isdigit():  # Check for nonnumeric or negative rise rate
            badRows.append(row)
        elif row > 0 and np.diff(data['Alt'])[row-1] <= 0:  # Check for stable or decreasing altitude (removes rise rate = 0)
            badRows.append(row)
        else:
            for col in range(data.shape[1]):  # Iterate through every cell in row
                if data.iloc[row, col] == 999999.0:  # This value is GRAWMET's version of NA
                    badRows.append(row)  # Remove row if 999999.0 is found
                    break

    if len(badRows) > 0:
        print("Dropping "+str(len(badRows))+" rows containing unusable data")
        data = data.drop(data.index[badRows])  # Actually remove any necessary rows
    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]

    # Get rid of extraneous columns that won't be used for further analysis
    essentialData = ['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Lat.', 'Long.']
    data = data[essentialData]

    return data  # return cleaned pandas data frame


def readFromData(file, path):
    # FUNCTION PURPOSE: Find launch time and pbl height in a profile file, or return default
    #                   values if not found. In particular, PBL height has to be written into
    #                   the profile by hand or by running companion software (CalculatePBL.py)
    #
    # INPUTS:
    #   file: The filename of the data file to read
    #   path: The path (absolute or relative) to the file
    #
    # OUTPUTS:
    #   launchDateTime: datetime.datetime object containing the UTC date and time of launch
    #   pblHeight: Number in meters representing PBL height



    # Establish default values, in case not contained in profile
    launchDateTime = datetime.datetime.now()
    pblHeight = 1500

    # Open and investigate the file
    f = open(os.path.join(path, file), 'r')
    for line in f:  # Iterate through file, line by line

        # If line has expected beginning, try to get datetime from file
        if line.rstrip() == "Flight Information:":
            try:
                dateTimeInfo = f.readline().split()
                dateTimeInfo = ' '.join(dateTimeInfo[2:6] + [dateTimeInfo[8]])
                launchDateTime = datetime.datetime.strptime(dateTimeInfo, '%A, %d %B %Y %H:%M:%S')
            except:
                # If an error is encountered, print a statement to the console and continue
                print("Error reading flight time info, defaulting to present")

        # If line has expected beginning, try to get PBL info
        if line.rstrip() == "PBL Information:":
            try:
                pblHeight = float(f.readline().split()[3])
            except:
                # If an error is encountered, print a statement to the console and continue
                print("Error reading flight PBL info, defaulting to 1500 meters")

    f.close()  # Need to close opened file

    # Return values from profile, or default values if not found
    return launchDateTime, pblHeight


########## PERFORMING ANALYSIS ##########


def interpolateData(data, spatialResolution, pblHeight, launchDateTime):
    # FUNCTION PURPOSE: Interpolate to create a Pandas DataFrame for the flight as a uniform
    #                   spatial grid, with datetime.datetime objects in the time column
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   spatialResolution: Desired length (in meters) between rows of data, must be a positive integer
    #   pblHeight: The height above ground in meters of the PBL
    #   launchDateTime: A datetime.datetime object containing the launch date and time in UTC
    #
    # OUTPUTS:
    #   data: Pandas DataFrame containing the time [s], altitude [m], temperature [deg C],
    #           pressure [hPa], wind speed [m/s], wind direction [deg], latitude [decimal deg],
    #           and longitude [decimal deg] of the radiosonde flight


    # First, filter data to remove sub-PBL data
    data = data[ (data['Alt'] - data['Alt'][0]) >= pblHeight]

    # Now, interpolate to create spatial grid, not temporal

    # Create index of heights with 1 meter spatial resolution
    heightIndex = pd.DataFrame({'Alt': np.arange(min(data['Alt']), max(data['Alt']))})
    # Right merge data with index to keeping all heights
    data = pd.merge(data, heightIndex, how="right", on="Alt")
    # Sort data by height for interpolation
    data = data.sort_values(by=['Alt'])

    # Use pandas built in interpolate function to fill in NAs
    # Linear interpolation appears the most trustworthy, but more testing could be done
    missingDataLimit = 999  # If 1 km or more missing data in a row, leave the NAs in place
    data = data.interpolate(method="linear", limit=missingDataLimit)

    # If NA's remain, missingDataLimit was exceeded
    if data.isnull().values.any():
        # Print event to console
        print("Found more than "+str(missingDataLimit)+" meters of consecutive missing data, quitting analysis.")
        # Return empty data frame, which means quit analysis
        return pd.DataFrame()

    data.reset_index(drop=True, inplace=True)  # Reset data frame index to [0,1,2,...,nrow]

    # Create index according to desired spatial resolution
    keepIndex = np.arange(0, len(data['Alt']), spatialResolution)
    data = data.iloc[keepIndex, :]  # Keep data according to index, lose the rest of the data

    data.reset_index(drop=True, inplace=True)  # Reset data frame index to [0,1,2,...,nrow]

    # Convert times from seconds since launch to UTC datetime.datetime objects
    times = data['Time'].copy()  # Make a copy of the column to stop warnings about inadvertent copying
    for n in range(len(times)):  # Iterate through time, turning times into datetime objects
        times[n] = launchDateTime + datetime.timedelta(seconds=float(times[n]))  # Add flight time to launch start
    data['Time'] = times  # Assign copy back to original data column

    return data  # Return pandas data frame


def waveletTransform(data, spatialResolution, wavelet):
    # FUNCTION PURPOSE: Perform the continuous wavelet transform on wind speed components and temperature
    #
    # INPUTS:
    #   data: Pandas DataFrame containing flight information
    #   spatialResolution: Length (in meters) between rows of data
    #   wavelet: String containing the name of the wavelet to use for the transformation. Based on
    #               Zink & Vincent (2001) and Murphy et. al (2014), this should be 'MORLET'
    #
    # OUTPUTS:
    #   results: Dictionary containing the power surface (|U|^2 + |V|^2), the wavelet transformed
    #               surfaces U, V, and T (zonal wind speed, meridional wind speed, and temperature
    #               in celsius), the wavelet scales and their corresponding fourier wavelengths,
    #               the cone of influence and the reconstruction constant from Torrence & Compo (1998)


    # u and v (zonal & meridional) components of wind speed
    u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
    v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

    # Subtract rolling mean (assumed to be background wind)
    # Up next, try a 2-3 order polynomial fit instead and see if there's a big difference
    N = int(len(data['Alt']) / 4)
    # Also, figure out what min_periods is really doing and make a reason for picking a good value
    rMean = pd.Series(u).rolling(window=N, min_periods=int(N/2), center=True).mean()
    u = u - rMean
    rMean = pd.Series(v).rolling(window=N, min_periods=int(N/2), center=True).mean()
    v = v - rMean
    rMean = pd.Series(data['T']).rolling(window=N, min_periods=int(N / 2), center=True).mean()
    t = data['T'] - rMean

    # In preparation for wavelet transformation, define variables
    # From Torrence & Compo (1998)
    padding = 1  # Pad the data with zeros to allow convolution to edge of data
    scaleResolution = 0.125/8  # This controls the spacing in between scales
    smallestScale = 2 * spatialResolution  # This number is the smallest wavelet scale
    howManyScales = 10/scaleResolution  # This number is how many scales to compute
    # Check Zink & Vincent section 3.2 par. 1 to see their scales/wavelengths

    # Lay groundwork for inversions, outside of looping over local max. in power surface
    # Derived from Torrence & Compo (1998) Equation 11 and Table 2
    constant = scaleResolution * np.sqrt(spatialResolution) / (0.776 * np.pi**0.25)

    # Now, do the actual wavelet transform using library from Torrence & Compo (1998)
    print("Performing wavelet transform on U... (1/3)", end='')  # Console output, to be updated
    coefU, periods, scales, coi = continuousWaveletTransform(u, spatialResolution, pad=padding, dj=scaleResolution, s0=smallestScale, mother=wavelet)  # Continuous morlet wavelet transform
    print("\rPerforming wavelet transform on V... (2/3)", end='')  # Update to keep user informed
    coefV, periods, scales, coi = continuousWaveletTransform(v, spatialResolution, pad=padding, dj=scaleResolution, s0=smallestScale, mother=wavelet)  # Continuous morlet wavelet transform
    print("\rPerforming wavelet transform on T... (3/3)", end='')  # Final console output for wavelet transform
    coefT, periods, scales, coi = continuousWaveletTransform(t, spatialResolution, pad=padding, dj=scaleResolution, s0=smallestScale, mother=wavelet)  # Continuous morlet wavelet transform


    # Power surface is sum of squares of u and v wavelet transformed surfaces
    power = abs(coefU) ** 2 + abs(coefV) ** 2  # abs() gets magnitude of complex number


    # Divide each column by sqrt of the scales so that it doesn't need to be done later to invert wavelet transform
    for col in range(coefU.shape[1]):
        coefU[:, col] = coefU[:, col] / np.sqrt(scales)
        coefV[:, col] = coefV[:, col] / np.sqrt(scales)
        coefT[:, col] = coefT[:, col] / np.sqrt(scales)

    results = {
        'power': power,
        'coefU': coefU,
        'coefV': coefV,
        'coefT': coefT,
        'scales': scales,
        'wavelengths': periods,
        'coi': coi,
        'constant': constant
    }

    return results  # Dictionary of wavelet-transformed surfaces


def findPeaks(power):

    # UI console output to keep user informed
    print("\nSearching for local maxima in power surface", end='')

    # Find and return coordinates of local maximums
    cutOff = 0.05  # Disregard maximums less than cutOff
    margin = 10  # Disregard maximums less than margin from image border, must be pos integer
    # Finds local maxima based on cutOff, margin
    peaks = peak_local_max(power, threshold_rel=cutOff, exclude_border=margin)

    # Currently debugging and testing this sort
    # Update 7/23, not needed for now
    #peaks = [x for _, x in sorted(zip(power[peaks[:,0], peaks[:,1]], peaks))]

    print()  # Newline for next console output

    return np.array(peaks)  # Array of coordinate arrays


def displayProgress(peaks, length):
    # Console output to keep user up to date
    print("\rTracing and analyzing peak " + str(length - len(peaks) + 1) + "/" + str(length), end='')


def findPeakSquare(power, peak):
    # Create boolean mask, initialized as False
    region = np.zeros(power.shape, dtype=bool)

    powerLimit = 0.25 * power[peak[0], peak[1]]

    row = power[peak[0], :]
    col = power[:, peak[1]]

    rowMins = np.array(argrelextrema(row, np.less))
    rowMins = np.append(rowMins, np.where(row <= powerLimit))
    rowMins = np.sort(np.append(rowMins, [0, peak[1], power.shape[1]-1]))
    cols = np.arange( rowMins[np.where(rowMins == peak[1])[0]-1], rowMins[np.where(rowMins == peak[1])[0]+1] + 1).tolist()

    colMins = np.array(argrelextrema(col, np.less))
    colMins = np.append(colMins, np.where(col <= powerLimit))
    colMins = np.sort(np.append(colMins, [0, peak[0], power.shape[0]-1]))
    rows = np.arange(colMins[np.where(colMins == peak[0])[0] - 1][0], colMins[np.where(colMins == peak[0])[0] + 1][0] + 1).tolist()

    region[np.ix_(rows, cols)] = True

    return region


def findPeakRegion(power, peak):
    # Create boolean mask, initialized as False
    region = np.zeros(power.shape, dtype=bool)
    region[peak[0], peak[1]] = True

    # Find cut-off power level, based on height of peak
    relativePowerLevel = 0.65  # Empirically determined parameter, to be adjusted
    absolutePowerLevel = power[peak[0], peak[1]] * relativePowerLevel
    # Find all the contours at cut-off level
    contours = find_contours(power, absolutePowerLevel)

    # Loop through contours to find the one surrounding the peak
    for contour in contours:

        # If the contour runs into multiple edges, skip as it's not worth trying
        if contour[0, 0] != contour[-1, 0] and contour[0, 1] != contour[-1, 1]:
            continue

        # Use matplotlib.path.Path to create a path
        p = path.Path(contour, closed=True)
        # Check to see if the peak is inside the closed loop of the contour path
        if p.contains_points([[peak[0], peak[1]]]):

            # If contour is not closed (includes an edge), close it manually
            if not (contour[0, :] == contour[-1, :]).all():  # Doesn't end where it starts, must be open

                # Get all points in between ends of contour

                # Assume positive directions for x and y iterations
                rowDir = 0.25
                colDir = 0.25
                # If not, change step to a negative
                if contour[0, 0] < contour[-1, 0]:
                    rowDir = -0.25
                if contour[0, 1] < contour[-1, 1]:
                    colDir = -0.25

                # List the rows and columns to iterate over
                rows = np.arange(contour[-1, 0], contour[0, 0] + rowDir, rowDir)
                cols = np.arange(contour[-1, 1], contour[0, 1] + colDir, colDir)

                # Finally, put index in proper format [ [row, col], [row2, col2] ]
                index = [[x, y] for x in rows for y in cols]
                index = np.array(index)

                # Add all points onto the end of contour
                contour = np.concatenate((contour, index), axis=0)

            # If it is, set the boundary path to True
            region[contour[:, 0].astype(int), contour[:, 1].astype(int)] = True

            # Then fill in the contour to create mask surrounding peak
            region = binary_fill_holes(region)

        # The method is now done, so return region
        return region

    # If for some reason the method couldn't isolate a region surrounding the peak,
    # set the peak itself to True so that it will be removed from list of peaks
    region[peak[0], peak[1]] = True
    # And return the almost empty mask
    return region


def removePeaks(region, peaks):
    # Remove local maxima that have already been traced from peaks list
    toRem = []  # Empty index of peaks to remove
    # Iterate through list of peaks
    for n in range(len(peaks)):
        if region[peaks[n][0], peaks[n][1]]:  # If peak in region,
            toRem.append(n)  # add peak to removal index
    peaks = [ value for (i, value) in enumerate(peaks) if i not in set(toRem) ]  # Then remove those peaks from peaks list
    return np.array(peaks)  # Return shortened list of peaks


def updatePlotter(region, plotter):
    # Copy the peak estimate to a plotting map
    plotter[region] = True

    return plotter  # Return plotting boolean mask


def invertWaveletTransform(region, wavelets):
    # Invert the wavelet transform in traced region

    uTrim = wavelets.get('coefU').copy()
    uTrim[np.invert(region)] = 0  # Trim U based on region
    # Sum across columns of U, then multiply by constant
    uTrim = np.multiply(uTrim.sum(axis=0), wavelets.get('constant'))

    # Do the same with V
    vTrim = wavelets.get('coefV').copy()
    vTrim[np.invert(region)] = 0
    vTrim = np.multiply( vTrim.sum(axis=0), wavelets.get('constant') )

    # Again with T
    tTrim = wavelets.get('coefT').copy()
    tTrim[np.invert(region)] = 0
    tTrim = np.multiply( tTrim.sum(axis=0), wavelets.get('constant') )

    # Declare results in dictionary
    results = {
        'uTrim': uTrim,
        'vTrim': vTrim,
        'tTrim': tTrim
    }

    return results  # Dictionary of trimmed inverted U, V, and T


def getParameters(data, wave, spatialResolution, region, waveAltIndex, wavelength, identifier):

    # Get index across the altitudes of the wave, for use later
    waveAlts = np.nonzero(region.sum(axis=0))

    # Calculate the wind variance of the wave
    windVariance = np.abs(wave.get('uTrim')) ** 2 + np.abs(wave.get('vTrim')) ** 2

    # Get rid of values below half-power, per Murphy (2014)
    uTrim = wave.get('uTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    vTrim = wave.get('vTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    tTrim = wave.get('tTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]

    # Get rid of unneeded variables, to save memory and improve performance
    del windVariance
    del wave

    # Seperate imaginary/real parts
    vHilbert = vTrim.copy().imag
    uvComp = [uTrim.copy(), vTrim.copy()]
    uTrim = uTrim.real
    vTrim = vTrim.real

    # Potential temperature
    pt = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)  # kelvin

    # Stokes parameters, still need some verification
    I = np.mean(uTrim ** 2) + np.mean(vTrim ** 2)
    D = np.mean(uTrim ** 2) - np.mean(vTrim ** 2)
    P = np.mean(2 * uTrim * vTrim)
    Q = np.mean(2 * uTrim * vHilbert)
    degPolar = np.sqrt(D ** 2 + P ** 2 + Q ** 2) / I

    # Check the covariance to perform additional filtering

    # Tests, I need to figure out why these make sense
    if np.abs(P) < 0.05 or np.abs(Q) < 0.05 or degPolar < 0.5 or degPolar > 1.0:
        return {}

    theta = 0.5 * np.arctan2(P, D)  # What is arctan2() and what makes it different from arctan()?


    # Classic 2x2 rotation matrix
    rotate = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    uvComp = np.dot(rotate, uvComp)  # Rotate so u and v components parallel/perpendicular to propogation direction

    #Alternative method that yields very similar results is axialRatio = np.abs(1 / np.tan(0.5 * np.arcsin(Q / (degPolar * I))))
    axialRatio = np.linalg.norm(uvComp[0]) / np.linalg.norm(uvComp[1])  # From Murphy (2014) Table 1, also referenced in Zink & Vincent

    # Make hodograph plot, for debugging
    #plt.scatter(uTrim, vTrim, marker='o', color='b')
    #plt.xlabel('U')
    #plt.ylabel('V')
    #plt.title('Hodograph of Traced Peak')
    #plt.show()

    gamma = np.mean(uvComp[0] * np.conj(tTrim)) / np.sqrt(np.mean(np.abs(uvComp[0]) ** 2) * np.mean(np.abs(tTrim) ** 2))
    if np.angle(gamma) < 0:
        theta = theta + np.pi

    coriolisF = np.abs( 2 * 7.2921 * 10 ** (-5) * np.sin(np.mean(data['Lat.']) * 180 / np.pi) )

    intrinsicF = coriolisF * axialRatio

    bvF2 = 9.81 / pt * np.gradient(pt, spatialResolution)  # Brunt-vaisala frequency squared???
    bvMean = np.mean(np.array(bvF2)[waveAlts])  # Mean of bvF2 across region

    if not np.sqrt(bvMean) > intrinsicF > coriolisF:
        return {}

    plt.figure()
    plt.scatter(uTrim, vTrim, c='blue')
    plt.xlabel("Zonal windspeed [m/s]")
    plt.ylabel("Meridional windspeed [m/s]")
    plt.title("Radiosonde Data Collected by MSGC, 2020")
    plt.savefig("C:/Users/12069/Documents/Eclipse2020/Presentation/Python Images/Comparison Hodographs/" +
                identifier + "_Peak_" + str(data['Alt'][waveAltIndex]) + "_" + str(wavelength) + "_ContourMethod.png")
    plt.close()


    # Values that I should output are:
    # Intrinsic frequency
    # Ground based frequency
    # Periods for above frequencies
    # Propagation direction
    # Altitude
    # Horizontal phase speed
    # Vertical wavelength

    # Vertical wavenumber [1/m]
    m = 2 * np.pi / wavelength
    # Horizontal wavenumber [1/m]
    kh = np.sqrt(((coriolisF ** 2 * m ** 2) / bvMean) * (intrinsicF ** 2 / coriolisF ** 2 - 1))
    # Intrinsic vertical wave velocity [m/s]
    intrinsicVerticalGroupVel = - (1 / (intrinsicF * m)) * (intrinsicF ** 2 - coriolisF ** 2)
    # Same [1/m]
    #zonalWaveNumber = kh * np.sin(theta)
    # Same [1/m]
    #meridionalWaveNumber = kh * np.cos(theta)
    # Same [m/s]
    intrinsicVerticalPhaseSpeed = intrinsicF / m
    # Same [m/s]
    intrinsicHorizPhaseSpeed = intrinsicF / kh
    # Same [m/s]
    intrinsicZonalGroupVel = kh * np.sin(theta) * bvMean / (intrinsicF * m ** 2)
    # Same [m/s]
    intrinsicMeridionalGroupVel = kh * np.cos(theta) * bvMean / (intrinsicF * m ** 2)
    # Same [m/s]
    intrinsicHorizGroupVel = np.sqrt(intrinsicZonalGroupVel ** 2 + intrinsicMeridionalGroupVel ** 2)
    # Horizontal wavelength [m]
    lambda_h = 2 * np.pi / kh
    # Altitude of wave peak
    altitudeOfDetection = data['Alt'][waveAltIndex]
    # Get latitude at index
    latitudeOfDetection = data['Lat.'][waveAltIndex]
    # Get longitude at index
    longitudeOfDetection = data['Long.'][waveAltIndex]
    # Get flight time at index
    timeOfDetection = data['Time'][waveAltIndex]

    # Assemble wave properties into dictionary
    waveProp = {
        'Altitude [km]': altitudeOfDetection / 1000,
        'Latitude [deg]': latitudeOfDetection,
        'Longitude [deg]': longitudeOfDetection,
        'Date and Time [UTC]': timeOfDetection,
        'Vertical wavelength [km]': (2 * np.pi / m) / 1000,
        'Horizontal wavelength [km]': lambda_h / 1000,
        'Angle of wave [deg]': theta * 180 / np.pi,
        'Axial ratio [no units]': axialRatio,
        'Intrinsic vertical group velocity [m/s]': intrinsicVerticalGroupVel,
        'Intrinsic horizontal group velocity [m/s]': intrinsicHorizGroupVel,
        'Intrinsic vertical phase speed [m/s]': intrinsicVerticalPhaseSpeed,
        'Intrinsic horizontal phase speed [m/s]': intrinsicHorizPhaseSpeed,
        'Degree of Polarization [no units]': degPolar,
        'Covariance between parallel and phase shifted perpendicular wind speed [m^2(s^-2)]': Q
    }

    return waveProp  # Dictionary of wave characteristics
