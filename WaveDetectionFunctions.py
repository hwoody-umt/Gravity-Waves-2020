########## IMPORT LIBRARIES AND FUNCTIONS ##########

import numpy as np  # Numbers (like pi) and math
import matplotlib.pyplot as plt  # Easy plotting
import pandas as pd  # Convenient data formatting, and who doesn't want pandas
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import os  # File reading and input
from io import StringIO  # Used to run strings through input/output functions
from scipy import interpolate  # Used for PBL calculations
import pywt  # Library PyWavelets, for wavelet transforms
from skimage.feature import peak_local_max  # Find local max
from scipy.ndimage.morphology import binary_fill_holes  # Help surround local max
import datetime  # Turning time into dates

########## PBL AND STABILITY CALCULATIONS ##########

def pblri(pt, u, v, hi):
    # This function calculates richardson number. It then
    # searches for where Ri(z) is near 0.25 and interpolates to get the height
    # z where Ri(z) = 0.25.
    #
    # INPUTS: write what these are eventually
    #
    # OUTPUTS: PBL height based on RI

    g = 9.81  # m/s/s
    ri = (pt - pt[0]) * hi * g / (pt * (u ** 2 + v ** 2))
    # This equation is right according to
    # https://www.researchgate.net/figure/Profile-of-potential-temperature-MR-and-Richardson-number-calculated-from-radiosonde_fig4_283187927
    # https://resy5.iket.kit.edu/RODOS/Documents/Public/CD1/Wg2_CD1_General/WG2_RP97_19.pdf

    # vt = vt[0:len(vt)-1]
    # ri = (np.diff(vpt) * np.diff(hi) * g / abs(vt)) / (np.diff(u) ** 2 + np.diff(v) ** 2)
    # print(ri)
    # Richardson number. If surface wind speeds are zero, the first data point
    # will be an inf or NAN.

    # Interpolate between data points
    riCutOff = 0.33
    f = interpolate.UnivariateSpline(hi, ri - riCutOff, s=0)
    #plt.plot(ri, hi)
    #plt.plot(f(hi) + riCutOff, ri)
    #plt.plot([0.33] * 2, plt.ylim())
    #plt.xlabel("RI")
    #plt.ylabel("Height above ground [m]")
    #plt.axis([-10, 20, 0, 5000])
    #plt.show()

    # Return heights where interpolation crosses riCutOff = 0.25
    # Need a way to pick which one is the right one... there are many
    if len(f.roots()) == 0:
        return [0]
    return f.roots()[0]

def pblpt(Alt, pot, hi):
    maxhidx = np.argmax(Alt)
    pth = pot[10:maxhidx]
    upH = hi[10:maxhidx]
    topH = 3800
    height3k = []
    for i, H in enumerate(upH):
        if H >= topH:
            break
        height3k.append(H)
    pt3k = pth[0:i]
    dp = np.gradient(pt3k, height3k)
    maxpidx = np.argmax(dp)
    pbl_potential_temperature = Alt[maxpidx]
    return pbl_potential_temperature

def pblsh(hi, rvv):
    maxhidx = np.argmax(hi)
    q = rvv / (1 + rvv)
    qh = q[10:maxhidx]
    upH = hi[10:maxhidx]
    topH = 3800
    height3k = []
    for i, H in enumerate(upH):
        if H >= topH:
            break
        height3k.append(H)
    qh3k = qh[0:i]
    dp = np.gradient(qh3k, height3k)
    minpix = np.argmin(dp)
    pbl_sh = hi[minpix]
    return pbl_sh

def pblvpt(vpt, hi):
    vptCutOff = vpt[1]
    f = interpolate.UnivariateSpline(hi, vpt - vptCutOff, s=0)
    #plt.plot(vpt, hi)
    #plt.plot(f(hi) + vptCutOff, hi)
    #plt.plot([vpt[1]] * 2, plt.ylim())
    #plt.axis([300, 400, 0, 3000])
    #plt.xlabel("VPT")
    #plt.ylabel("Height above ground [m]")
    #plt.show()
    return 0

def layerStability(hi, pot):
    ds = 1
    # du = 0.5 doesn't seem to be used... ?
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

def calculatePBL(data):

    #Calculate necessary variables

    # Height above ground
    hi = data['Alt'] - data['Alt'][0]  # meters
    # Epsilon, from?
    epsilon = 0.622  # unitless constant
    # Vapor pressure
    e = 6.1121 * np.exp((18.678 - (data['T'] / 234.84)) * (data['T'] / (257.14 + data['T']))) * data['Hu']  # hPa
    # Water vapor mixing ratio
    rvv = (epsilon * e) / (data['P'] - e)  # unitless
    # Potential temperature
    pot = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)  # kelvin
    # virtual potential temperature
    vpt = pot * ((1 + (rvv / epsilon)) / (1 + rvv))  # kelvin
    # absolute virtual temperature
    vt = (data['T'] + 273.15) * ((1 + (rvv / epsilon)) / (1 + rvv))  # kelvin
    # u and v (east & north?) components of wind speed
    u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
    v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

    # Get three different PBL height estimations
    #pblHeightRI = pblri(pot, u, v, hi)
    pblHeightRI = 1500  # Fix above line, get Hannah's code and figure the shit out
    #pblHeightVPT = pblvpt(vpt, hi)
    pblHeightVPT = 1500  # Same here
    pblHeightPT = pblpt(data['Alt'], pot, hi)
    pblHeightSH = pblsh(hi, rvv)

    # Make preliminary analysis plots, dependent on user input showPlots
    #if showPlots:
    #    drawPlots(data['Alt'], data['T'], data['Dewp.'], pblHeightRI)#,pblHeightPT,pblHeightSH)

    # Calculate which PBL height to use
    # pblHeight = max(pblHeightRI), pblHeightPT, pblHeightSH)
    pblHeight = np.max(np.array([pblHeightRI, pblHeightVPT, pblHeightPT, pblHeightSH]))
    print("Calculated PBL height of " + str(pblHeight))
    print(layerStability(hi, pot))
    return pblHeight  # Return best guess for pbl height

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
            userInput = ""
    if lower(userInput) == "y":
        return True
    else:
        return False

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
        print("\nFile "+file+" is not a text file, ending analysis.")
        return pd.DataFrame(), 0  # Empty data frame means end analysis


    # Open and investigate the file
    contents = ""
    isProfile = False  # Check to see if this is a GRAWMET profile
    launchDateTime = datetime.datetime.now()
    f = open(os.path.join(path, file), 'r')
    print("\nOpening file "+file+":")
    for line in f:  # Iterate through file, line by line
        if line.rstrip() == "Flight Information:":
            dateTimeInfo = f.readline().split()
            dateTimeInfo = ' '.join(dateTimeInfo[2:6] + [dateTimeInfo[8]])
            launchDateTime = datetime.datetime.strptime(dateTimeInfo, '%A, %d %B %Y %H:%M:%S')

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

    return ( data, launchDateTime )  # return cleaned pandas data frame and time of launch

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
        print("Found more than "+str(missingDataLimit)+" consecutive missing data, quitting analysis.")
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

    # List of wavelet scales to iterate over
    scaleResolution = 10  # How far apart to choose list of scales
    scales = np.arange(10, 4000, scaleResolution)  # How should we pick the scales???
    # Above range seems to be good (via visual inspection), but a logarithmic resolution I think is the way to go... to fix later
    # Possibly look at literature for frequency, then convert to scale and figure it out?

    # Now, do the actual wavelet transform
    print("Performing wavelet transform on U... (1/3)", end='')  # Console output, to be updated
    (coefU, freq) = pywt.cwt(u, scales, wavelet, spatialResolution)  # Continuous morlet wavelet transform
    print("\b\b\b\b\b\b\b\b\b\bV... (2/3)", end='')  # Update to keep user from getting too bored
    (coefV, freq) = pywt.cwt(v, scales, wavelet, spatialResolution)
    print("\b\b\b\b\b\b\b\b\b\bT... (3/3)")  # Final command line update for wavelet analysis
    (coefT, freq) = pywt.cwt(data['T'], scales, wavelet, spatialResolution)

    # Power surface is sum of squares of u and v wavelet transformed surfaces
    power = abs(coefU) ** 2 + abs(coefV) ** 2  # abs() gets magnitude of complex number

    # Lay groundwork for inversions, outside of local max. loop
    # Magic constant hypothetically from Torrence and Compo, Table 2 & Eqn 11
    magicConstant = scaleResolution * np.sqrt(spatialResolution) / (0.776 * np.pi ** 0.25)  # Investigate, figure this out
    # Divide each column by sqrt(scales)
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
        'constant': magicConstant
    }

    #np.save("wavelets.npy", results)
    #data.to_csv("dataex.csv", index=False)

    return results  # Dictionary of wavelet-transformed surfaces

def findPeaks(power):

    # Find and return coordinates of local maximums
    cutOff = 0.4  # Disregard maximums less than cutOff * imageMax
    margin = 10  # Disregard maximums less than margin from image border, must be pos integer
    distance = 10  # Disregard maximums less than distance away from each other, must be pos integer
    # Finds local maxima based on distance, cutOff, margin
    peaks = peak_local_max(power, min_distance=distance, threshold_rel=cutOff, exclude_border=margin)

    return peaks  # Array of coordinate arrays

# Older peak finding code is here
# def findPeaks(power):
#
#     # Find and return coordinates of local maximums
#     cutOff = 0.50  # Disregard maximums less than cutOff * imageMax
#     margin = 10  # Disregard maximums less than margin from image border, must be pos integer
#     distance = 25  # Disregard maximums less than distance away from each other, must be pos integer
#     # Finds local maxima based on distance, cutOff, margin
#     peaks = peak_local_max(power, min_distance=distance, threshold_rel=cutOff, exclude_border=margin)
#
#     return peaks  # Array of coordinate arrays

def displayProgress(peaks, length):

    # Console output to keep user from getting too bored
    print("\rTracing and analyzing peak " + str(length - len(peaks)) + "/" + str(length), end='')

def searchNearby(row, col, region, power, powerLimit, tol):
    for r in range(row-tol, row+tol+1):
        for c in range(col-tol, col+tol+1):
            try:
                if not region[r,c] and powerLimit < power[r,c]:# <= power[row, col]+1:
                    region[r,c] = True
                    region = searchNearby(r,c, region, power, powerLimit, tol)
            except IndexError:
                pass
    return region

def findPeakRegion(power, peak):
    region = np.zeros(power.shape, dtype=bool)
    region[peak[0], peak[1]] = True

    powerLimit = 0.75 * power[peak[0], peak[1]]

    tolerance = 2

    try:
        region = searchNearby(peak[0], peak[1], region, power, powerLimit, tolerance)
        region = binary_fill_holes(region)
    except RecursionError:
        try:
            region = searchNearby(peak[0], peak[1], region, power, powerLimit, tolerance)
            region = binary_fill_holes(region)
        except RecursionError:
            pass

    return region

# Older region finding code is here
# def searchPowerSurface(X, Y, row, col, rMod, cMod, power, powerLimit, initialCall):
#     # This method needs hella comments, get to it eventually
#     onEdge = False
#     iRow = row
#     iCol = col
#     iRMod = rMod
#     iCMod = cMod
#     while power[row, col] > powerLimit:
#         if (row == 0 or row == power.shape[0] - 1) and rMod is not 0:
#             if onEdge:
#                 return X, Y
#             rMod = 0
#             onEdge = True
#             if cMod == 0:
#                 if initialCall:
#                     cMod = 1
#                     X, Y = searchPowerSurface(X, Y, iRow, iCol, iRMod, iCMod, power, powerLimit, False)
#                 else:
#                     cMod = -1
#
#         if (col == 0 or col == power.shape[1] - 1) and cMod is not 0:
#             if onEdge:
#                 return X, Y
#
#             cMod = 0
#             onEdge = True
#             if rMod == 0:
#                 if initialCall:
#                     rMod = 1
#                     X, Y = searchPowerSurface(X, Y, iRow, iCol, iRMod, iCMod, power, powerLimit, False)
#                 else:
#                     rMod = -1
#         row += rMod
#         col += cMod
#
#     Y.append([row])
#     X.append([col])
#     return X, Y
#
# def findPeakRegion(power, peak):
#     # This method needs hella comments too, and cleaning up in the looks department
#
#     # Initialize regions to False
#     region = np.zeros(power.shape, dtype=bool)
#
#     # Get peak coordinates
#     r = peak[0]
#     c = peak[1]
#
#
#     powerLimit = 0.25  # Percentage of peak height to form lower barrier
#     powerLimit = powerLimit * power[r, c]
#
#     X, Y = ( [], [] )
#
#     for rMod in [-1,0,1]:
#         for cMod in [-1,0,1]:
#             if (rMod is not 0) or (cMod is not 0):
#                 X, Y = searchPowerSurface(X, Y, r, c, rMod, cMod, power, powerLimit, True)
#
#     # Formulate and solve the least squares problem ||Ax - b ||^2
#     A = np.hstack( (np.power(X,2), np.multiply(X, Y), np.power(Y,2), X, Y) )
#     b = np.ones_like(X)
#     x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
#
#     if x[1] ** 2 - 4 * x[0] * x[2] >= 0:  # Fit is hyperbolic/parabolic, not elliptical
#         region[r, c] = True  # Remove peak from list
#         return region  # Didn't find anything
#
#     X_coord, Y_coord = np.meshgrid(range(region.shape[1]), range(region.shape[0]))
#     region = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
#
#     region = ( region.astype(int) > 0 )
#
#     return region  # Boolean mask showing region surrounding peak

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
    #print("Nonzero elements of U: " + str(np.count_nonzero(uTrim)))
    uTrim[np.invert(region)] = 0  # Trim U based on region
    #print("Nonzero elements of U: " + str(np.count_nonzero(uTrim)))
    # Sum across columns of U, then multiply by mysterious constant
    uTrim = np.multiply([sum(x) for x in uTrim.T.tolist()], wavelets.get('constant'))
    #print("Nonzero elements of U: " + str(np.count_nonzero(uTrim)))
    # Do the same with V
    vTrim = wavelets.get('coefV').copy()
    vTrim[np.invert(region)] = 0
    vTrim = np.multiply( [ sum(x) for x in vTrim.T.tolist() ], wavelets.get('constant') )
    # Again with T
    tTrim = wavelets.get('coefT').copy()
    tTrim[np.invert(region)] = 0
    tTrim = np.multiply( [ sum(x) for x in tTrim.T.tolist() ], wavelets.get('constant') )

    results = {
        'uTrim': uTrim,
        'vTrim': vTrim,
        'tTrim': tTrim,
        'scales': wavelets.get('scales')
    }

    return results  # Dictionary of trimmed inverted U, V, and T

def getParameters(data, wave, spatialResolution, region):
    # Find wind variance, why? I don't know.
    windVariance = np.abs(wave.get('uTrim')) ** 2 + np.abs(wave.get('vTrim')) ** 2  # What is this, why do we care?

    # Why do we do this? I have no idea...
    uTrim = wave.get('uTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    vTrim = wave.get('vTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]
    tTrim = wave.get('tTrim').copy()[windVariance >= 0.5 * np.max(windVariance)]

    # Seperate imaginary/real parts
    vHilbert = vTrim.imag
    uvComp = [uTrim, vTrim]
    uTrim = uTrim.real
    vTrim = vTrim.real

    # I'll figure out where to put this later
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
        # print("\nShit it's bad.")
        # print(np.abs(P))
        # print(np.abs(Q))
        # print(degPolar)
        return {}

    theta = 0.5 * np.arctan2(P, D)  # What the hell?
    axialRatio = np.abs(1 / np.tan(0.5 * np.arcsin(Q / (degPolar * I))))  # What is this?

    # Classic 2x2 rotation matrix
    rotate = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    uvComp = np.dot(rotate, uvComp)  # Rotate so u and v components parallel/perpendicular to propogation direction
    gamma = np.mean(uvComp[0] * np.conj(tTrim)) / np.sqrt(np.mean(np.abs(uvComp[0]) ** 2) * np.mean(np.abs(tTrim) ** 2))
    if np.angle(gamma) < 0:
        theta = theta + np.pi
    coriolisF = 2 * 7.2921 * 10 ** (-5) * np.sin(np.mean(data['Lat.']) * 180 / np.pi)
    intrinsicF = coriolisF * axialRatio
    bvF2 = 9.81 / pt * np.gradient(pt, spatialResolution)  # Brunt-vaisala frequency squared???
    bvMean = np.mean(np.array(bvF2)[np.nonzero([sum(x) for x in region.T])])  # Mean of bvF2 across region

    if not np.sqrt(bvMean) > np.abs(intrinsicF) > np.abs(coriolisF):
        # print("\nTriple shit")
        # print(np.sqrt(bvMean))
        # print(intrinsicF)
        # print(coriolisF)
        return {}

    # Vertical wavenumber [1/m]
    m = 2 * np.pi / np.mean(np.array(1.03 * wave.get('scales'))[np.nonzero([sum(x) for x in region])])
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
    # Horizontal wavelength? [m]
    lambda_h = 2 * np.pi / kh
    # Just average altitude of wave [m]
    altitudeOfDetection = np.mean(np.array(data['Alt'])[np.nonzero([sum(x) for x in region.T])])
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
        'Angle of wave [deg]': theta,
        'Axial ratio [no units]': axialRatio,
        'Intrinsic vertical group velocity [m/s]': intrinsicVerticalGroupVel,
        'Intrinsic horizontal group velocity [m/s]': intrinsicHorizGroupVel,
        'Intrinsic vertical phase speed [m/s]': intrinsicVerticalPhaseSpeed,
        'Intrinsic horizontal phase speed [m/s]': intrinsicHorizPhaseSpeed,
        'Degree of Polarization [WHAT IS THIS???]': degPolar,
        'Mysterious parameter Q [who knows]': Q
    }

    return waveProp  # Dictionary of wave characteristics
