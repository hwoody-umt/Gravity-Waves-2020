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
#import cmath  # Complex numbers... I'm still not sure how these work in the analysis
import csv

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
    return f.roots()

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
    return str(pbl_potential_temperature)

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

def pblvpt(pot, rvv, vpt, hi):
    pot = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)
    epsilon = 0.622  # epsilon, unitless constant
    virtcon = 0.61
    rvv = (epsilon * e) / (data['P'] - e)  # unitless
    vpt = pot * (1 + (virtcon * rvv))
    
    vptCutOff = vpt[1]
    g = interpolate.UnivariateSpline(hi, vpt - vptCutOff, s=0)
    plt.plot(vpt, hi)
    plt.plot(g(hi)+vptCutOff, hi)
    plt.plot([vpt[1]] * 2, plt.ylim())
    plt.axis([300, 400, 0, 3000])
    plt.xlabel("VPT")
    plt.ylabel("Height above ground [m]")
    plt.show()
    
    rootuno = []
    rootdos = []
    
    if len(g.roots()) == 0:
        return [0]
    for H in g.roots():
        if H >= 1000:
            rootuno.append(H)
    for J in rootuno:
        if J <= 3000:
            rootdos.append(J)
    return max(rootdos)


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
    
def largestpbl (pblHeightRI, pblHeightVPT, pblHeightPT, pblHeightSH):
    listofheights = [str(pblHeightRI), str(pblHeightPT), str(pblHeightSH), str(pblHeightVPT)]
    Output = sorted(listofheights, key = lambda x:float(x))
    return Output[-1]

def drawPlots(alt, t, td, pblHeightRI, pblHeightVPT):  # , pblHeightPT, pblHeightSH):
    print("Displaying data plots")

    # Plot radiosonde path
    plt.plot(data['Long.'], data['Lat.'])
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
    pblHeightRI = pblri(vpt, vt, pot, u, v, hi)
    pblHeightVPT = pblvpt(pot, rvv, vpt, hi)
    pblHeightPT = pblpt(hi, pot)
    pblHeightSH = pblsh(hi, rvv)
    pblLargestHeight = largestpbl(pblHeightRI, pblHeightVPT, pblHeightPT, pblHeightSH)
    print(layerStability(hi, pot))
    print("HIGHEST PBL HEIGHT CALCULATED " + str(pblLargestHeight))

    # Make preliminary analysis plots, dependent on user input showPlots
    #if showPlots:
    #    drawPlots(data['Alt'], data['T'], data['Dewp.'], pblHeightRI)#,pblHeightPT,pblHeightSH)

    # Calculate which PBL height to use
    # pblHeight = max(pblHeightRI), pblHeightPT, pblHeightSH)
    pblHeight = np.max([pblHeightRI, pblHeightVPT, pblHeightPT, pblHeightSH])
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
    if  not file.endswith(".txt"):
        print("\nFile "+file+" is not a text file, ending analysis.")
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

    return data  # return cleaned pandas data frame

########## PERFORMING ANALYSIS ##########

def interpolateData(data, spatialResolution, pblHeight):

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
        print("Found more than "+missingDataLimit+" consecutive missing data, quitting analysis.")
        return pd.DataFrame()

    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]
    keepIndex = np.arange(0, len(data['Alt']), spatialResolution)  # Index altitude by spatialRes
    data = data.iloc[keepIndex, :]  # Keep data according to index
    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]

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

    return results  # Dictionary of wavelet-transformed surfaces

def findPeaks(power):

    # Find and return coordinates of local maximums
    print("Isolating local maximums... ", end='')
    cutOff = 0.50  # Disregard maximums less than cutOff * imageMax
    margin = 10  # Disregard maximums less than margin from image border, must be pos integer
    distance = 25  # Disregard maximums less than distance away from each other, must be pos integer
    # Finds local maxima based on distance, cutOff, margin
    peaks = peak_local_max(power, min_distance=distance, threshold_rel=cutOff, exclude_border=margin)

    return peaks  # Array of coordinate arrays

def searchNearby(iR, iC, power, regions, cutOff, tol):
    list1 = np.arange(iR - tol, iR + tol)
    list2 = np.arange(iC - tol, iC + tol)
    for r in list1:
        for c in list2:
            # Find super fast way to do this check... try except clause?
            # if (r in range(regions.shape[0])) and (c in range(regions.shape[1])):  # out of bounds :P
            if (not regions[r, c]) and cutOff < power[r, c] <= power[iR, iC]:
                regions[r, c] = True
                regions = searchNearby(r, c, power, regions, cutOff, tol)
    return regions

def displayProgress(peaks, count):

    # Console output to keep user from getting too bored
    if count == 0:  # First, need to print precursor to numbers
        print("tracing peak "+str(count + 1)+"/"+str(len(peaks)), end='')
    elif count > 10:  # Two digit number
        print("\b\b\b\b\b" + str(count + 1) + "/" + str(len(peaks)), end='')
    else:  # One digit number
        print("\b\b\b\b"+str(count + 1)+"/"+str(len(peaks)), end='')
    # Increment counter
    return count + 1  # Return incremented counter

def findPeakRegion(power, peak):
    # Initialize regions to False
    regions = np.zeros(power.shape, dtype=bool)

    # Get peak coordinates
    row = peak[0]
    col = peak[1]

    # Recursively check power surface downhill until hitting low power limit
    powerLimit = 0.5  # Percentage of peak height to form lower barrier
    flexibility = 5  # Number of indices algorithm is allowed to reach, 1 means can only flag adjacent cells
    regions = searchNearby(row, col, power, regions, powerLimit * power[row, col], flexibility)

    # Fill in local maximums that were surrounded but ignored
    regions = binary_fill_holes(regions)

    return regions  # Boolean mask showing region surrounding peak

def removePeaks(region, peaks):
    # Remove local maxima that have already been traced from peaks list
    toRem = []  # Empty index of peaks to remove
    # Iterate through list of peaks
    for n in range(len(peaks)):
        if region[peaks[n][0], peaks[n][1]]:  # If peak in region,
            toRem.append(n)  # add peak to removal index
    peaks = np.delete(peaks, toRem)  # Then remove those peaks from peaks list

    return peaks  # Return shortened list of peaks

def updatePlotter(region, plotter):
    # Copy the peak estimate to a plotting map

    # Iterate over cells in region
    for row in range(region.shape[0]):
        for col in range(region.shape[1]):
            # Add True cells in region to plotting map
            if region[row, col]:
                plotter[row, col] = True

    return plotter  # Return plotting boolean mask

def invertWaveletTransform(region, wavelets):
    # Invert the wavelet transform in traced region

    uTrim = wavelets.get('coefU')
    uTrim[np.invert(region)] = 0  # Trim U based on region
    # Sum across columns of U, then multiply by mysterious constant
    uTrim = np.multiply([sum(x) for x in uTrim.T.tolist()], wavelets.get('constant'))
    # Do the same with V
    vTrim = wavelets.get('coefV')
    vTrim[np.invert(region)] = 0
    vTrim = np.multiply( [ sum(x) for x in vTrim.T.tolist() ], wavelets.get('constant') )
    # Again with T
    tTrim = wavelets.get('coefT')
    tTrim[np.invert(region)] = 0
    tTrim = np.multiply( [ sum(x) for x in tTrim.T.tolist() ], wavelets.get('constant') )

    results = {
        'uTrim': uTrim,
        'vTrim': vTrim,
        'tTrim': tTrim,
        'scales': wavelets.get('scales')
    }

    return results  # Dictionary of trimmed inverted U, V, and T

def getParameters(data, wave, spatialResolution):

    # Find wind variance, why? I don't know.
    windVariance = wave.get('uTrim') ** 2 + wave.get('vTrim') ** 2  # What is this, why do we care?

    # Why do we do this? I have no idea...
    index = windVariance >= 0.5 * np.max(windVariance)
    uTrim = wave.get('uTrim')[ index ]
    vTrim = wave.get('vTrim')[ index ]
    tTrim = wave.get('tTrim')[ index ]

    # Seperate imaginary/real parts
    vHilbert = vTrim.imag
    uvComp = [ uTrim, vTrim ]
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
    degPolar = np.sqrt( D**2 + P**2 + Q**2 ) / I

    # Tests, I need to figure out why these make sense
    if abs(P) < 0.05 or abs(Q) < 0.05 or degPolar < 0.5 or degPolar > 1.0:
        print("\nShit it's bad.")
        print(abs(P))
        print(abs(Q))
        print(degPolar)
        return {}

    theta = 0.5 * np.arctan2(P, D)  # What the hell?
    axialRatio = abs( 1 / np.tan( 0.5 * np.arcsin( Q / ( degPolar * I ) ) ) )  # What is this?

    # Classic 2x2 rotation matrix
    rotate = [ [np.cos(theta), np.sin(theta) ], [-np.sin(theta), np.cos(theta) ] ]
    uvComp = np.dot( rotate, uvComp )  # Rotate so u and v components parallel/perpendicular to propogation direction
    gamma = np.mean( uvComp[0] * np.conj(tTrim) ) / np.sqrt( np.mean( abs(uvComp[0])**2 ) * np.mean( abs(tTrim)**2 ) )
    if np.angle(gamma) < 0:
        theta = theta + np.pi
    coriolisF = 2 * 7.2921 * 10**(-5) * np.sin( np.mean(data['Lat.']) * 180 / np.pi )
    intrinsicF = coriolisF * axialRatio
    bvF2 = 9.81 / pt * np.gradient(pt, spatialResolution)  # Brunt-vaisala frequency squared???
    bvMean = np.mean( np.array(bvF2)[ np.nonzero( [ sum(x) for x in region.T ] ) ] )  # Mean of bvF2 across region

    if not np.sqrt(bvMean) > abs(intrinsicF) > abs(coriolisF):
        print("\nTriple shit")
        print(np.sqrt(bvMean))
        print(intrinsicF)
        print(coriolisF)
        return {}

    # Vertical wavenumber [1/m]
    m = 2 * np.pi / np.mean( np.array(1.03 * wave.get('scales'))[ np.nonzero( [ sum(x) for x in region ] ) ] )
    # Horizontal wavenumber [1/m]
    kh = np.sqrt( ( ( coriolisF**2 * m**2 ) / bvMean ) * ( intrinsicF**2 / coriolisF**2 - 1) )
    # I don't really know [m/s]
    intrinsicVerticalGroupVel = - (1 / (intrinsicF * m)) * (intrinsicF**2 - coriolisF**2)
    # Same [1/m]
    zonalWaveNumber = kh * np.sin(theta)
    # Same [1/m]
    meridionalWaveNumber = kh * np.cos(theta)
    # Same [m/s]
    intrinsicVerticalPhaseSpeed = intrinsicF / m
    # Same [m/s]
    intrinsicHorizPhaseSpeed = intrinsicF / kh
    # Same [m/s]
    intrinsicZonalGroupVel = zonalWaveNumber * bvMean / (intrinsicF * m**2)
    # Same [m/s]
    intrinsicMeridionalGroupVel = meridionalWaveNumber * bvMean / (intrinsicF * m**2)
    # Same [m/s]
    intrinsicHorizGroupVel = np.sqrt(intrinsicZonalGroupVel**2 + intrinsicMeridionalGroupVel**2)
    # Horizontal wavelength? [m]
    lambda_h = 2 * np.pi / kh
    # Just average altitude of wave [m]
    altitudeOfDetection = np.mean( np.array(data['Alt'])[ np.nonzero( [ sum(x) for x in region.T ] ) ])
    # Get index of mean altitude
    detectionIndex = np.where(np.min(np.abs(data['Alt'] - altitudeOfDetection)))
    # Get latitude at index
    latitudeOfDetection = np.array(data['Lat.'])[detectionIndex]
    # Get longitude at index
    longitudeOfDetection = np.array(data['Long.'])[detectionIndex]
    # Assemble wave properties into dictionary
    waveProp = {
        'Altitude [km]': altitudeOfDetection/1000,
        'Latitude [deg]': latitudeOfDetection,
        'Longitude [deg]': longitudeOfDetection,
        'Vertical wavelength [km]': (2*np.pi/m)/1000,
        'Horizontal wavelength [km]': lambda_h/1000,
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


########## ACTUAL RUNNING CODE ##########

# First, get applicable user input.
userInput = getAllUserInput()
# Then, iterate over files in data directory
for file in os.listdir( userInput.get('dataSource') ):
    # Import and clean the data, given the file path
    data = cleanData( file, userInput.get('dataSource') )
    if not data.empty:
        pblHeight = calculatePBL( data )
        spatialResolution = 5  # meters in between uniformly distributed data points, must be pos integer
        data = interpolateData( data, spatialResolution, pblHeight )
        if not data.empty:
            data.to_csv("dataex.csv")
            if userInput.get('showPlots'):
                drawPlots( data['Alt'], data['T'], data['Dewp.'], pblHeight )
            # Get the stuff... comment better later!
            wavelets = waveletTransform( data, spatialResolution, 'cmor2-6')  # Use morlet wavelet
            # Find local maximums in power surface
            peaks = findPeaks( wavelets.get('power') )

            # Numpy array for plotting purposes
            plotter = np.zeros( wavelets.get('power').shape, dtype=bool )
            count = 0  # Counter to output progress

            waves = []  # Empty list, to contain wave characteristics

            # Iterate over local maximums to identify wave characteristics
            while len(peaks > 0):
                # Output progress to console and increment counter
                count = displayProgress( peaks, count )
                # Identify the region surrounding the peak
                region = findPeakRegion( wavelets.get('power'), peaks[count - 1] )
                # Update list of peaks that have yet to be analyzed
                peaks = removePeaks( region, peaks )
                # Update plotting mask
                plotter = updatePlotter( region, plotter )

                # Get inverted regional maximums
                wave = invertWaveletTransform( region, wavelets )
                # Get wave parameters
                parameters = getParameters( data, wave, spatialResolution )
                if parameters:
                    #print(parameters)  # Just for debugging right now...
                    waves = waves.append(parameters)

            # Save waves data here, if saveData boolean is true
            print(waves)
            # Also, build nice output plot
            plt.figure()
            plt.imshow( wavelets.get('power'), extent=[data['Alt'][0:-1]/1000, (1 / pywt.scale2frequency('morl',wavelets.get('scales')))[0:-1] ] )
            cb = plt.colorbar()
            plt.contour(data['Alt'] / 1000, 1 / pywt.scale2frequency('morl',wavelets.get('scales')), plotter, colors='red')
            # plt.contour(data['Alt'], freq, power, colors='red')
            plt.xlabel("Altitude [km]")
            plt.ylabel("Period [m]")
            plt.title("Ummmmm")
            cb.set_label("Power [m^2/s^2]")

            if userInput.get('saveData'):
                plt.savefig(userInput.get('savePath') + "/" + file[0:-4] + "_power_surface.png")
            if userInput.get('showPlots'):
                plt.show()
            plt.close()
            print("Finished analysis.")

########## FINISHED ANALYSIS ##########
print("\nAnalyzed all files in folder "+userInput.get('dataSource')+"/")

