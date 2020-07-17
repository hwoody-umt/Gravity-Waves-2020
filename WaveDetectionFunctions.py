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

########## PBL AND STABILITY CALCULATIONS ##########

def pblRI(vpt, u, v, hi):
    # The RI method looks for a Richardson number between 0 and 0.25 above a height of 800 and below a height of 3000
    # The function will then return the highest height of a "significant RI"
    # NOTE: these boundaries will need to be changed if you are taking data from a night time flight,
    # where the PBL can be lower than 800m
    # or a flight near the equator where the PBL can exceed a height of 3000m

    # Support for this method and the equation used can be found at the following link:
    # https://resy5.iket.kit.edu/RODOS/Documents/Public/CD1/Wg2_CD1_General/WG2_RP97_19.pdf
    # https://watermark.silverchair.com/1520-0469(1979)036_0012_teorwa_2_0_co_2.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsMwggK_BgkqhkiG9w0BBwagggKwMIICrAIBADCCAqUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMtQBGOhb8zchwxViIAgEQgIICduC_f9w94ccDO9Nz1u73Ti7uOmXyjo_dLzL6LsXhu0-0uMAxTRsrPuPu_aCgyt4vyLVccC1OeRc9KR5npTEGstzVFFZs-vFNNs8Bl78f1K5jOhlAT9DYH3oSp3vdEM763kaZDV_1mc-8QzJORohbeGB1YOu4TbqYd70ZoJCS59yKO7emrSfcVVdQIWNOQ6PoT4ONeDowOCXCIgv4WBO-ul9fKAuA217EvXIh3-5o_SGj-SuMO30ktr8htOstvD_dC36eB3efxJ9l2MyDwvurUAO4CfJBgpaCKAg4af8LeljpmlXbFgkB7_jQyVXYvdfZNxvjAmp72Nbn6x_qjRc3TMhrhzw4R0ZtwjF9IjfDz-zolAwDPZ_PALKP-HE-M-Zi7q9hRd6XxDsjVOINTpZ07apgpT0ssX58uU3aPAiWDZnEInwz2-r_b_6KJHABRFWj4GYmW34v35nQz_xCo20S3MRQ-Lh7CiiwIAvkchNIfpScUI11Kz7Hd8gLsVqQ7r8fp4iWbgc4NEkS2gRkj8XEIqdvvFyCLLPo6bs_20iVtyEuGuwWQM3fYbpiS38iqth9LFcx7suDYUbMd1GbrYR3gdbvr9KKLohN6-rCJV-8rxIDOqraPxewIJyOckPHEaQ5Ek1Q1FEahweLE3HDgz93DnDQHoHYrmDU0gmsvDRqtxVRnqVf95d3V5DQNom8MFPEZiRdv7Vb8-2BQq_GMYEXZrv0FeKVr40HLSRy5Kc4qXZBR97XjN04AEyJ-umhyrb5DuzQdksk2T5WTXIIlx3DmZXLYY5Ond0cXDhOjGh7A6sPiJ2jVPTEzwSdwXUtpnMdxdpsFX6GhQ

    g = 9.81  # m/s/s
    ri = (((vpt - vpt[0]) * hi * g) / (vpt[0] * (u ** 2 + v ** 2)))
    #  Check for positive RI <= 0.25 and height between 800 and 3000 meters
    index = [ 0 <= a <= 0.25 and 800 <= b <= 3000 for a, b in zip(ri, hi)]


    #plt.plot(ri, hi)
    #plt.xlim(-100, 100)
    #plt.plot([0.25,0.25], [0,30000])
    #plt.show()

    if np.sum(index) > 0:  # If there are results, return them
        return np.max( hi[index] )

    # Otherwise, interpolate to find height

    # Trim to range we're interested in
    index = [800 <= n <= 3000 for n in hi]
    hi = hi[index]
    ri = ri[index]

    # Interpolate, returning either 800 or 3000 if RI doesn't cross 0.25
    return np.interp(0.25, ri, hi)


def pblPT(hi, pot):
    # The potential temperature method looks at a vertical gradient of potential temperature, the maximum of this gradient
    # ie: Where the change of potential temperature is greatest
    # However, due to the nature of potential temperature changes you will need to interpret this data from the graph
    # produced by this function. In this case you are looking for a maxium on the POSITIVE side of the X=axis around the
    # height that makes sense for your predicted PBL height
    # NOTE: arrays at the top of the function start at index 10 to avoid noise from the lower indexes.

    # Support for this method can be found at the following link:
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD013680

    # High and low height limits for the PBL
    topH = 2000
    lowH = 800

    # Trim potential temperature and height to within specified heights
    height = [i for i in hi if lowH <= i <= topH]
    pt = [p for p, h in zip(pot, hi) if lowH <= h <= topH]

    dp = np.gradient(pt, height)  # creates a gradient of potential temperature and height

    #plt.plot(dp, height3k)  # creates the plot you will need to read to determine the PBL Height
    #plt.ylim(800, 2000)  # Change this if there is reason to believe PBL may be higher than 2000, or lower than 800
    #plt.xlabel("Gradient of PT")
    #plt.ylabel("Height above ground in meters")
    #plt.show()
    #return getUserInputNum("Please enter the PBL height according to this plot:")

    # Return height of maximum gradient
    return np.array(height)[dp == np.max(dp)]


def pblSH(hi, rvv):
    # The specific humidity method looks at a vertical gradient of specific humidity, the minimum of this gradient
    # ie: where the change in gradient is the steepest in negative direction
    # However, due to the nature of specific humidity changes you will need to interpret this data from the graph
    # produced by this function. In this case you are looking for a maxium on the NEGATIVE side of the X=axis around the
    # height that makes sense for your predicted PBL height
    # NOTE: arrays at the top of the function start at index 10 to avoid noise from the lower indexes.

    # Support for this method can be found at the following link:
    # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD013680

    q = rvv / (1 + rvv)  # equation for specific humidity

    # High and low height limits for the PBL
    topH = 2000
    lowH = 800

    # Trim potential temperature and height to within specified heights
    height = [i for i in hi if lowH <= i <= topH]
    q = [q for q, h in zip(q, hi) if lowH <= h <= topH]

    dp = np.gradient(q, height)  # creates a gradient of potential temperature and height

    #plt.plot(dp, height3k)  # creates the plot you will need to read to determine the PBL Height
    #plt.ylim(800, 2000)  # Change this if there is reason to believe the PBL may be higher than 2000 or lower than 800
    #plt.xlabel("Gradient of Specific Humidity")
    #plt.ylabel("Height above ground in meters")
    #plt.show()
    #return getUserInputNum("Please enter the PBL height according to this plot:")

    # Return height at maximum gradient
    return np.array(height)[dp == np.max(dp)]


def pblVPT(vpt, hi):
    # The Virtual Potential Temperature (VPT) method looks for the height at which VPT is equal to the VPT at surface level
    # NOTE: The VPT may equal VPT[0] in several places, so the function is coded to return the highest height where
    # these are equal

    # Supoort for this method can be found at the following link:
    # https://www.mdpi.com/2073-4433/6/9/1346/pdf

    roots = np.interp(vpt[0], vpt, hi)  # Finds heights at which

    return roots


def layerStability(hi, pot):
    # This function looks through potential temperature data to determine layer stability into 3 catergories
    # NOTE: It is best to choose the higest PBL calculation unless the methods produce PBL Heights more than 300m
    # apart. Also, when a stable boundary layer is detected, reject a PBL that is above 2000m, as these are often
    # night-time layers and a PBL near 2000m does not make sense

    ds = 1
    try:
        diff = [pot[i] for i in range(len(pot)) if hi[i] >= 150]
        diff = diff[0] - pot[0]
    except:
        return "Unable to detect layer stability, possibly due to corrupt data"

    if diff < -ds:
        return "Detected convective boundary layer"
    elif diff > ds:
        return "Detected stable boundary layer"
    else:
        return "Detected neutral residual layer"


def drawPlots(alt, t, td, pblHeightRI, pblHeightVPT):  # , pblHeightPT, pblHeightSH):
    print("Displaying data plots")

    # Plot radiosonde path
    #plt.plot(data['Long.'], data['Lat.'])
    plt.ylabel("Latitude [degrees]")
    plt.xlabel("Longitude [degrees]")
    plt.title("Radiosonde Flight Path")
    plt.show()

    # Plot pbl estimates
    pblHeightRI += alt[0]  # Convert height to altitude
    # pblHeightSH += alt[0]
    plt.plot(t, alt, label="Temperature")
    plt.plot(td, alt, label="Dewpoint")
    # plt.plot(plt.get_xlim(),[pblHeightPT] * 2, label="PT Method")
    plt.plot(plt.xlim(), [pblHeightRI] * 2, label="RI Method")
    plt.plot(plt.xlim(), [pblHeightVPT] * 2, label="VPT Method")
    plt.axis([-80, 20, 1000, 3500])
    # plt.plot(t,[pblHeightSH] * 2, label="SH Method")
    plt.title('PBL Calculations')
    plt.xlabel("Temperature [deg. C]")
    plt.ylabel("Altitude [m]")
    plt.legend()
    plt.show()

########## USER INPUT SECTION ##########
def getUserInputFile(prompt):
    print(prompt)
    userInput = ""
    while not userInput:
        userInput = input()
        if not os.path.isdir(userInput):
            print("Please enter a valid directory:")
            userInput = ""
    return userInput

def getUserInputTF(prompt):
    print(prompt+" (Y/N)")
    userInput = ""
    while not userInput:
        userInput = input()
        if lower(userInput) != "y" and lower(userInput) != "n":
            print("Please enter a valid answer (Y/N):")
            userInput = ""
    if lower(userInput) == "y":
        return True
    else:
        return False

def getUserInputNum(prompt):
    print(prompt)
    userInput = ""
    while not userInput:
        userInput = input()
        if not userInput.isdigit():
            print("Please enter a valid integer:")
            userInput = ""
    return int(userInput)

def getAllUserInput():
    dataSource = getUserInputFile("Enter path to data input directory: ")
    showPlots = getUserInputTF("Do you want to display plots for analysis?")
    saveData = getUserInputTF("Do you want to save the output data?")
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
    print("Path to output data: "+savePath+"/\n")

    # Build a dictionary to return values
    results = {
        'dataSource': dataSource,
        'showPlots': showPlots,
        'saveData': saveData
    }
    if saveData:
        results.update( {'savePath': savePath })

    return results

########## DATA INPUT SECTION ##########

def cleanData(file, path):
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
    # Open and investigate the file

    # Establish default values, in case not contained in profile
    launchDateTime = datetime.datetime.now()
    pblHeight = 1500

    f = open(os.path.join(path, file), 'r')
    for line in f:  # Iterate through file, line by line

        if line.rstrip() == "Flight Information:":
            try:
                dateTimeInfo = f.readline().split()
                dateTimeInfo = ' '.join(dateTimeInfo[2:6] + [dateTimeInfo[8]])
                launchDateTime = datetime.datetime.strptime(dateTimeInfo, '%A, %d %B %Y %H:%M:%S')
            except:
                print("Error reading flight time info, defaulting to present")

        if line.rstrip() == "PBL Information:":
            try:
                pblHeight = float(f.readline().split()[3])
            except:
                print("Error reading flight PBL info, defaulting to 1500 meters")

    f.close()  # Need to close opened file

    return launchDateTime, pblHeight

########## PERFORMING ANALYSIS ##########

def interpolateData(data, spatialResolution, pblHeight, launchDateTime):

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
    missingDataLimit = 999  # If 1 km or more missing data in a row, leave the NAs
    data = data.interpolate(method="linear", limit=missingDataLimit)

    if data.isnull().values.any():  # More than 1000 meters missing data
        print("Found more than "+str(missingDataLimit)+" meters of consecutive missing data, quitting analysis.")
        return pd.DataFrame()

    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]
    keepIndex = np.arange(0, len(data['Alt']), spatialResolution)  # Index altitude by spatialRes
    data = data.iloc[keepIndex, :]  # Keep data according to index
    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]

    times = data['Time'].copy()  # Make a copy of the column to stop warnings about inadvertent copying
    for n in range(len(times)):  # Iterate through time, turning times into datetime objects
        times[n] = launchDateTime + datetime.timedelta(seconds=float(times[n]))  # Add flight time to launch start
    data['Time'] = times  # Assign copy back to original data column

    return data  # Return pandas data frame

def waveletTransform(data, spatialResolution, wavelet):

    # u and v (east & north?) components of wind speed
    u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
    v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

    # Subtract rolling mean (assumed to be background wind)
    # Window calculation here is kinda sketchy, so investigate
    # N = max( altitude extent / height sampling / 4, 11) in Tom's code
    N = 1000  # We'll go with 1 km for now and then come back to see what's up later
    rMean = pd.Series(u).rolling(window=N, min_periods=1, center=True).mean()
    u = u - rMean
    rMean = pd.Series(v).rolling(window=N, min_periods=500, center=True).mean()
    v = v - rMean

    # In preperation for wavelet transformation, define variables
    # From Torrence & Compo (1998)
    padding = 1  # Pad the data with zeros to allow convolution to edge of data
    scaleResolution = 1/1500  # This is a scale thingamobober
    smallestScale = 100  # This number is the smallest wavelet scale
    #j1 = 10/dj  # This number is how many scales to compute

    # Lay groundwork for inversions, outside of local max. loop
    # Derived from Torrence & Compo, 1998, Equation 11 and Table 2
    constant = scaleResolution * np.sqrt(spatialResolution) / (0.776 * np.pi**0.25)

    # Now, do the actual wavelet transform
    print("Performing wavelet transform on U... (1/3)", end='')  # Console output, to be updated
    coefU, periods, scales, coi = continuousWaveletTransform(u, spatialResolution, pad=padding, dj=scaleResolution, s0=smallestScale, mother=wavelet)  # Continuous morlet wavelet transform
    print("\rPerforming wavelet transform on V... (2/3)", end='')  # Update to keep user informed
    coefV, periods, scales, coi = continuousWaveletTransform(v, spatialResolution, pad=padding, dj=scaleResolution, s0=smallestScale, mother=wavelet)  # Continuous morlet wavelet transform
    print("\rPerforming wavelet transform on T... (3/3)", end='')  # Final console update for wavelet transform
    coefT, periods, scales, coi = continuousWaveletTransform(data['T'], spatialResolution, pad=padding, dj=scaleResolution, s0=smallestScale, mother=wavelet)  # Continuous morlet wavelet transform


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
        'constant': constant
    }

    return results  # Dictionary of wavelet-transformed surfaces

def findPeaks(power):

    # UI console output to keep user informed
    print("\nSearching for local maxima in power surface", end='')

    # Find and return coordinates of local maximums
    cutOff = 0.25  # Disregard maximums less than cutOff * imageMax
    margin = 10  # Disregard maximums less than margin from image border, must be pos integer
    distance = 1  # Disregard maximums less than distance away from each other, must be pos integer
    # Finds local maxima based on distance, cutOff, margin
    peaks = peak_local_max(power, min_distance=distance, threshold_rel=cutOff, exclude_border=margin)

    print()  # Newline for next console output

    return peaks  # Array of coordinate arrays

def displayProgress(peaks, length):

    # Console output to keep user from getting too bored
    print("\rTracing and analyzing peak " + str(length - len(peaks) + 1) + "/" + str(length), end='')

def searchNearby(row, col, region, power, powerLimit, tol):
    for r in range(row-tol, row+tol+1):
        for c in range(col-tol, col+tol+1):
            try:
                if not region[r,c] and powerLimit < power[r,c] <= power[row, col]:
                    region[r,c] = True
                    region = searchNearby(r,c, region, power, powerLimit, tol)
            except IndexError:
                pass
    return region
#
# def findPeakRegion(power, peak):
#     region = np.zeros(power.shape, dtype=bool)
#     region[peak[0], peak[1]] = True
# #
#     powerLimit = 0.75 * power[peak[0], peak[1]]
# #
#     tolerance = 2
# #
#     try:
#         region = searchNearby(peak[0], peak[1], region, power, powerLimit, tolerance)
#         region = binary_fill_holes(region)
#     except RecursionError:
#         try:
#             region = searchNearby(peak[0], peak[1], region, power, powerLimit, tolerance)
#             region = binary_fill_holes(region)
#         except RecursionError:
#             pass
# #
#     return region

# Older region finding code is here
def searchPowerSurface(X, Y, row, col, rMod, cMod, power, powerLimit, initialCall):
    # This method needs hella comments, get to it eventually
    onEdge = False
    iRow = row
    iCol = col
    iRMod = rMod
    iCMod = cMod
    while power[row, col] > powerLimit:
        if (row == 0 or row == power.shape[0] - 1) and rMod is not 0:
            if onEdge:
                return X, Y
            rMod = 0
            onEdge = True
            if cMod == 0:
                if initialCall:
                    cMod = 1
                    X, Y = searchPowerSurface(X, Y, iRow, iCol, iRMod, iCMod, power, powerLimit, False)
                else:
                    cMod = -1
#
        if (col == 0 or col == power.shape[1] - 1) and cMod is not 0:
            if onEdge:
                return X, Y
#
            cMod = 0
            onEdge = True
            if rMod == 0:
                if initialCall:
                    rMod = 1
                    X, Y = searchPowerSurface(X, Y, iRow, iCol, iRMod, iCMod, power, powerLimit, False)
                else:
                    rMod = -1
        row += rMod
        col += cMod
#
    Y.append([row])
    X.append([col])
    return X, Y

# def findPeakRegion(power, peak):
#     region = np.zeros(power.shape, dtype=bool)
#     rows = 20
#     cols = 150
#
#     region[(peak[0]-rows):(peak[0]+rows), (peak[1]-cols):(peak[1]+cols)] = True
#
#     return region

def findPeakRegion(power, peak):
    # Create boolean mask, initialized as False
    region = np.zeros(power.shape, dtype=bool)

    # Find cut-off power level, based on height of peak
    relativePowerLevel = 0.5  # Empirically determined parameter, to be adjusted
    absolutePowerLevel = power[peak[0], peak[1]] * relativePowerLevel
    # Find all the contours at cut-off level
    contours = find_contours(power, absolutePowerLevel)

    # Loop through contours to find the one surrounding the peak
    for contour in contours:
        # Use matplotlib.path.Path to create a path
        p = path.Path(contour)
        # Check to see if the peak is inside the closed loop of the contour path
        if p.contains_points([[peak[0], peak[1]]]):
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
    return peaks  # Return shortened list of peaks

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

    # Add wavelengths (filtered according to region) for use in later analysis
    results.update({'wavelengths': wavelets.get('wavelengths')[np.nonzero(region.sum(axis=1))]})

    return results  # Dictionary of trimmed inverted U, V, and T

def getParameters(data, wave, spatialResolution, region):

    # Get index across the altitudes of the wave, for use later
    waveAlts = np.nonzero(region.sum(axis=0))

    # Calculate the wind variance of the wave
    windVariance = np.abs(wave.get('uTrim')) ** 2 + np.abs(wave.get('vTrim')) ** 2

    # Get rid of values below half-power, per Murphy
    uTrim = wave.get('uTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    vTrim = wave.get('vTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    tTrim = wave.get('tTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]

    # Get rid of unneeded variables, to save memory and improve performance
    wavelengths = wave.get('wavelengths')
    del windVariance
    del wave

    # Seperate imaginary/real parts
    vHilbert = vTrim.copy().imag
    uvComp = [uTrim.copy(), vTrim.copy()]
    uTrim = uTrim.real
    vTrim = vTrim.real

    # Potential temperature
    pt = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)  # kelvin

    # Straight from Tom's code, what are these and why do we do this?
    I = np.mean(uTrim ** 2) + np.mean(vTrim ** 2)
    D = np.mean(uTrim ** 2) - np.mean(vTrim ** 2)
    P = np.mean(2 * uTrim * vTrim)
    Q = np.mean(2 * uTrim * vHilbert)
    degPolar = np.sqrt(D ** 2 + P ** 2 + Q ** 2) / I

    # Tests, I need to figure out why these make sense
    if np.abs(P) < 0.05 or np.abs(Q) < 0.05 or degPolar < 0.5 or degPolar > 1.0:
        return {}

    theta = 0.5 * np.arctan2(P, D)  # What the hell?
    axialRatio = np.abs(1 / np.tan(0.5 * np.arcsin(Q / (degPolar * I))))  # What is this?

    # Classic 2x2 rotation matrix
    rotate = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    uvComp = np.dot(rotate, uvComp)  # Rotate so u and v components parallel/perpendicular to propogation direction

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

    # Vertical wavenumber [1/m]
    m = 2 * np.pi / np.mean(wavelengths)
    # Horizontal wavenumber [1/m]
    kh = np.sqrt(((coriolisF ** 2 * m ** 2) / bvMean) * (intrinsicF ** 2 / coriolisF ** 2 - 1))
    # I don't really know [m/s]
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
    # Just average altitude of wave [m]
    altitudeOfDetection = np.mean(np.array(data['Alt'])[waveAlts])
    # Get index of mean altitude
    detectionIndex = [np.min(np.abs(data['Alt'] - altitudeOfDetection)) == np.abs(data['Alt'] - altitudeOfDetection)]
    # Get latitude at index
    latitudeOfDetection = np.array(data['Lat.'])[tuple(detectionIndex)]
    # Get longitude at index
    longitudeOfDetection = np.array(data['Long.'])[tuple(detectionIndex)]
    # Get flight time at index
    timeOfDetection = np.array(data['Time'])[tuple(detectionIndex)]

    # Assemble wave properties into dictionary
    waveProp = {
        'Altitude [km]': altitudeOfDetection / 1000,
        'Latitude [deg]': latitudeOfDetection[0],
        'Longitude [deg]': longitudeOfDetection[0],
        'Date and Time [UTC]': timeOfDetection[0],
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
