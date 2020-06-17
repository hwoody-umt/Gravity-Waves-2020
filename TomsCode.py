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
import cmath  # Complex numbers... I'm still not sure how these work in the analysis

########## Function definitions, to be used later ##########

def pblri(vpt, vt, pt, u, v, hi):
    # This function calculates richardson number. It then
    # searches for where Ri(z) is near 0.25 and interpolates to get the height
    # z where Ri(z) = 0.25.
    #
    # INPUTS: write what these are eventually
    #
    # OUTPUTS: PBL height based on RI

    g = 9.81  # m/s/s
    ri = (pt - pt[0]) * hi * g / ( pt * (u ** 2 + v ** 2) )
    # This equation is right according to
    #https://www.researchgate.net/figure/Profile-of-potential-temperature-MR-and-Richardson-number-calculated-from-radiosonde_fig4_283187927
    #https://resy5.iket.kit.edu/RODOS/Documents/Public/CD1/Wg2_CD1_General/WG2_RP97_19.pdf

    #vt = vt[0:len(vt)-1]
    #ri = (np.diff(vpt) * np.diff(hi) * g / abs(vt)) / (np.diff(u) ** 2 + np.diff(v) ** 2)
    #print(ri)
    # Richardson number. If surface wind speeds are zero, the first data point
    # will be an inf or NAN.

    # Interpolate between data points
    riCutOff = 0.25
    f = interpolate.UnivariateSpline(hi, ri - riCutOff, s=0)
    #plt.plot(ri, hi)
    #plt.plot(f(hi)+riCutOff, hi)
    #plt.plot([0.25] * 2, plt.ylim())
    #plt.xlabel("RI")
    #plt.ylabel("Height above ground [m]")
    #plt.axis([-10, 20, 0, 5000])
    #plt.show()

    # Return heights where interpolation crosses riCutOff = 0.25
    # Need a way to pick which one is the right one... there are many
    if len(f.roots()) == 0:
        return [0]
    return f.roots()

def pblpt(hi, pot):
    # This function calculates PBL height based on potential temperature method
    maxhidx = max(hi)
    pth = pot[10:maxhidx]
    upH = hi[10:maxhidx]
    topH = 3500
    height3k = [i for i in upH if upH[i] <= topH]
    pt3k = [i for i in pth if upH[i] <= topH]
    dp3k = np.gradient(pt3k, height3k)
    maxdpidx = max(dp3k)
    return height3k * maxdpidx

def pblsh(hi, rvv):
    # This function calculates PBL height using another method - WHAT?
    maxhidx = max(hi)
    q = rvv/(1+rvv)
    qh = q[10:maxhidx]
    upH = hi[10:maxhidx]
    topH = 3500
    height3k = upH(upH<=topH)
    q3k = qh(upH<=topH)
    dq3k = np.gradient(q3k,height3k)
    dq = np.gradient(q,hi)
    mindpidx = min(dq3k)
    return height3k * mindpidx

def layerStability(hi, pot):
    ds = 1
    #du = 0.5 doesn't seem to be used... ?
    try:
        diff = [pot[i] for i in range(len(pot)) if hi[i] >= 150]
        diff = diff[0]-pot[0]
    except:
        return "Unable to detect layer stability, possibly due to corrupt data"

    if diff < -ds:
        return "Detected convective boundary layer"
    elif diff > ds:
        return "Detected stable boundary layer"
    else:
        return "Detected neutral residual layer"

def drawPlots(alt, t, td, pblHeightRI):#, pblHeightPT, pblHeightSH):
    print("Displaying data plots")

    # Plot radiosonde path
    plt.plot(data['Long.'], data['Lat.'])
    plt.ylabel("Latitude [degrees]")
    plt.xlabel("Longitude [degrees]")
    plt.title("Radiosonde Flight Path")
    plt.show()

    # Plot pbl estimates
    pblHeightRI += alt[0]  # Convert height to altitude
    #pblHeightPT += alt[0]
    #pblHeightSH += alt[0]
    plt.plot(t, alt, label="Temperature")
    plt.plot(td, alt, label="Dewpoint")
    #plt.plot(plt.get_xlim(),[pblHeightPT] * 2, label="PT Method")
    plt.plot(plt.xlim(), [pblHeightRI] * 2, label="RI Method")
    #plt.plot(t,[pblHeightSH] * 2, label="SH Method")
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
            userInput = ""
    if lower(userInput) == "y":
        return True
    else:
        return False


dataSource = getUserInputFile("Enter path to data input directory: ")
showPlots = getUserInputTF("Do you want to display plots for analysis?")
saveData = getUserInputTF("Do you want to save the output data?")
if saveData:
    savePath = getUserInputFile("Enter path to data output directory: ")
else:
    savePath = "\b\bNA"
# MATLAB code has lower and upper altitude cut-offs and latitude
# I've changed these to be read in from the data

# For debugging, print results
print("Running with the following parameters:")
print("Path to input data: ./"+dataSource+"/")
print("Display plots: "+str(showPlots))
print("Save data: "+str(saveData))
print("Path to output data: ./"+savePath+"/\n")

########## FILE RETRIEVAL SECTION ##########

# Need to find all txt files in dataSource directory and iterate over them

# However, I also want to check the GRAWMET software to see if it can output
# the profile in either a JSON or CSV file format, as that would likely be
# much easier.


for file in os.listdir(dataSource):
    if file.endswith(".txt"):

        #Used to fix a file reading error
        contents = ""
        #Check to see if this is a GRAWMET profile
        isProfile = False
        f = open(os.path.join(dataSource, file), 'r')
        print("\nOpening file "+file+":")
        for line in f:
            if line.rstrip() == "Profile Data:":
                isProfile = True
                contents = f.read()
                print("File contains GRAWMET profile data")
                break
        f.close()
        if not isProfile:
            print("File "+file+" is either not a GRAWMET profile, or is corrupted.")

        if isProfile:  # Read in the data and perform analysis

            # Fix a format that causes a table reading error
            contents = contents.replace("Virt. Temp", "Virt.Temp")
            contents = contents.split("\n")
            contents.pop(1)  # Remove units from temp file
            index = -1
            for i in range(0, len(contents)):  # Find beginning of footer
                if contents[i].strip() == "Tropopauses:":
                    index = i
            if index >= 0:  # Remove footer, if found
                contents = contents[:index]
            contents = "\n".join(contents)  # Reassemble string
            del index

            # Read in the data
            print("Constructing a data frame")
            data = pd.read_csv(StringIO(contents), delim_whitespace=True)
            del contents

            # Find the end of usable data
            badRows = []
            for row in range(data.shape[0]):
                if not str(data['Rs'].loc[row]).replace('.', '', 1).isdigit():  # Check for nonnumeric or negative rise rate
                    badRows.append(row)
                elif row > 0 and np.diff(data['Alt'])[row-1] <= 0:
                    badRows.append(row)
                else:
                    for col in range(data.shape[1]):
                        if data.iloc[row, col] == 999999.0:  # This value appears a lot and is obviously wrong
                            badRows.append(row)
                            break
            if len(badRows) > 0:
                print("Dropping "+str(len(badRows))+" rows containing unusable data")
            data = data.drop(data.index[badRows])
            data.reset_index(drop=True, inplace=True)
            ########## PERFORMING ANALYSIS ##########

            #Calculate variables needed for further analysis

            hi = data['Alt'] - data['Alt'][1]  # height above ground in meters
            epsilon = 0.622  # epsilon, unitless constant

            # vapor pressure
            e = 6.1121 * np.exp((18.678 - (data['T'] / 234.84)) * (data['T'] / (257.14 + data['T']))) * data['Hu']  # hPa

            # water vapor mixing ratio
            rvv = (epsilon * e) / (data['P'] - e)  # unitless

            # potential temperature
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
            #pblHeightPT = pblpt(hi, pot) needs some serious work
            #pblHeightSH = pblsh(hi, rvv) needs some serious work
            print("Calculated PBL height of "+str(pblHeightRI))#+", "+str(pblHeightPT)+", and "+str(pblHeightSH)+" meters")
            print(layerStability(hi, pot))

            # Make preliminary analysis plots, dependent on user input showPlots
            #if showPlots:
            #    drawPlots(data['Alt'], data['T'], data['Dewp.'], pblHeightRI)#,pblHeightPT,pblHeightSH)
            del epsilon, e, rvv, pot, vpt, vt

            # Which PBL height to use
            pblHeight = max(pblHeightRI)#, pblHeightPT, pblHeightSH)
            # Filter data to remove sub-PBL data
            data = data[hi >= pblHeight]
            del hi

            # Now, interpolate to create spatial grid, not temporal
            data = pd.merge(data, pd.DataFrame({'Alt': np.arange(min(data['Alt']), max(data['Alt']))}), how="right", on="Alt")
            data = data.sort_values(by=['Alt'])
            data = data.interpolate(method="linear", limit=999)
            tempBool = True
            if data.isnull().values.any():  # More than 1000 meters missing data
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH!!!!!!")
                tempBool = False

            spatialRes = 2  # meters between data points, must be pos integer
            data.reset_index(drop=True, inplace=True)
            keepIndex = np.arange(0, len(data['Alt']), 5)
            data = data.iloc[keepIndex, :]  # Lower spatial resolution to 5 meters
            data.reset_index(drop=True, inplace=True)

            # u and v (east & north?) components of wind speed
            u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
            v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

            # potential temperature
            pt = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)  # kelvin

            if showPlots & tempBool:
                drawPlots(data['Alt'], data['T'], data['Dewp.'], pblHeightRI)#,pblHeightPT,pblHeightSH)

            # Subtract rolling mean (assumed to be background)
            # Window calculation here is kinda sketchy, so investigate
            # N = max( altitude extent / height sampling / 4, 11) in Tom's code
            N = 1000  # We'll go with 1 km for now and then come back to see what's up later
            rMean = pd.Series(u).rolling(window=N, min_periods=1, center=True).mean()
            u = u - rMean
            rMean = pd.Series(v).rolling(window=N, min_periods=500, center=True).mean()
            v = v - rMean

            # Now, do the actual wavelet transform
            print("Performing wavelet transform on U...", end='')
            scaleRes = 10
            scales = np.arange(10, 4000, scaleRes)  # How should we pick the scales???
            # Possibly look at literature for frequency, then convert to scale and figure it out
            (coefU, freq) = pywt.cwt(u, scales, 'morl', spatialRes)  # Continuous morlet wavelet transform
            print("\b\b\b\bV...", end='')
            (coefV, freq) = pywt.cwt(v, scales, 'morl', spatialRes)
            print("\b\b\b\bT...")
            (coefT, freq) = pywt.cwt(data['T'], scales, 'morl', spatialRes)
            print("Done performing wavelet transform.")

            # Plotting and saving code
            power = coefU ** 2 + coefV ** 2

            # Lay groundwork for inversions, outside of local max. loop
            # Magic constant hypothetically from Torrence and Compo, Table 2 & Eqn 11
            magicConstant = scaleRes * np.sqrt(spatialRes) / (0.776 * np.pi ** 0.25)  # Investigate, figure this out
            # Divide each column by sqrt(scales)
            for col in range(coefU.shape[1]):
                coefU[:, col] = coefU[:, col] / np.sqrt(scales)
                coefV[:, col] = coefV[:, col] / np.sqrt(scales)
                coefT[:, col] = coefT[:, col] / np.sqrt(scales)

            # Now, look for power surface maximums, clip and invert!
            print("Isolating local maximums")
            cutOff = 0.50  # Disregard maximums less than cutOff * imageMax
            margin = 10  # Disregard maximums less than margin from image border, must be pos integer
            peaks = peak_local_max(power, min_distance=25, threshold_rel=cutOff, exclude_border=margin)


            def searchNearby(iR, iC, power, regions, cutOff, tol):
                list1 = np.arange(iR - tol, iR + tol)
                list2 = np.arange(iC - tol, iC + tol)
                for r in list1:
                    for c in list2:
                        if (r in range(regions.shape[0])) and (c in range(regions.shape[1])) and (not regions[r, c]) and cutOff < power[r, c] <= power[iR, iC]:
                            regions[r, c] = True
                            regions = searchNearby(r, c, power, regions, cutOff, tol)
                return regions


            plotter = np.zeros(power.shape, dtype=bool)
            count = 0
            while len(peaks > 0):

                # Console output to keep user from getting too bored
                if count == 0:
                    print("Tracing peak "+str(count + 1)+"/"+str(len(peaks)), end='')
                elif count > 10:
                    print("\b\b\b\b\b" + str(count + 1) + "/" + str(len(peaks)), end='')
                else:
                    print("\b\b\b\b"+str(count + 1)+"/"+str(len(peaks)), end='')

                # Initialize regions to False
                regions = np.zeros(power.shape, dtype=bool)
                # Get peak coordinates
                row = peaks[count][0]
                col = peaks[count][1]
                # Recursively check power surface downhill until hitting low power limit
                regions = searchNearby(row, col, power, regions, 0.5 * power[row, col], 5)
                # Fill in local maximums that were surrounded but ignored
                regions = binary_fill_holes(regions)

                # Now remove local maxima that have been traced from peaks list
                toRem = []
                for peak in peaks:
                    if regions[peak[0], peak[1]]:  # If peak in regions,
                        toRem.append(peak)  # add peak to removal index
                for peak in toRem:  # Then remove those peaks from peaks list
                    peaks.remove(peak)
                del toRem
                # Increment counter
                count += 1

                # Copy the peak estimate to a plotting map
                for row in range(regions.shape[0]):
                    for col in range(regions.shape[1]):
                        if regions[row, col]:
                            plotter[row, col] = True

                # Plot anyways, just for debugging
                plt.figure()
                plt.contourf(data['Alt'] / 1000, 1 / freq, power)
                cb = plt.colorbar()
                plt.contour(data['Alt'] / 1000, 1 / freq, regions, colors='red')
                # plt.contour(data['Alt'], freq, power, colors='red')
                plt.xlabel("Altitude [km]")
                plt.ylabel("Period [m]")
                plt.title("Ummmmm")
                cb.set_label("Power [m^2/s^2]")
                plt.show()

                # Now invert the wavelet transform in traced region
                # Trim to local max in question, then sum columns
                uTrim = [ sum(x) for x in (coefU[regions]).T ] * magicConstant
                vTrim = [ sum(x) for x in (coefV[regions]).T ] * magicConstant
                tTrim = [ sum(x) for x in (coefT[regions]).T ] * magicConstant
                windVariance = uTrim ** 2 + vTrim ** 2  # What is this, why do we care?

                # Now let's try to get wave properties...

                # Why do we do this? I have no idea...
                index = windVariance >= 0.5 * np.max(windVariance)
                uTrim = uTrim[ index ]
                vTrim = vTrim[ index ]
                tTrim = tTrim[ index ]
                # Seperate imaginary/real parts
                vHilbert = vTrim.imag
                uvComp = [ uTrim, vTrim ]
                uTrim = uTrim.real
                vTrim = vTrim.real

                # Straight from Tom's code, why???
                I = np.mean(uTrim ** 2) + np.mean(vTrim ** 2)
                D = np.mean(uTrim ** 2) - np.mean(vTrim ** 2)
                P = np.mean(2 * uTrim * vTrim)
                Q = np.mean(2 * uTrim * vHilbert)
                degPolar = np.sqrt( D**2 + P**2 + Q**2 ) / I

                # Tests, figure out what the hell is going on
                if np.abs(P) < 0.05 or np.abs(Q) < 0.05 or degPolar < 0.5 or degPolar > 1.0:
                    print("Well, shit. This didn't work at all.")
                else:
                    theta = 0.5 * np.arctan2(P,D)  # What the hell?
                    axialRatio = np.abs( np.cot( 0.5 * np.arcsin( Q / ( degPolar * I ) ) ) )
                    if axialRatio < 1:
                        print("Well, shit. This didn't work either")
                    else:
                        rotate = [ [np.cos(theta), np.sin(theta) ], [-np.sin(theta), np.cos(theta) ] ]
                        uvComp = np.multiply( rotate, uvComp )
                        gamma = np.mean( uvComp[0] * tTrim.conj ) / np.sqrt( np.mean( uvComp[0]**2 ) * np.mean( tTrim**2 ) )
                        if gamma.phase < 0:
                            theta = theta + np.pi
                        coriolisF = 2 * 7.2921 * 10**(-5) * np.sin( np.mean(data['Lat.']) * 180 / np.pi )
                        intrinsicF = coriolisF * axialRatio
                        bvF2 = 9.81 / pt * np.gradient(pt, spatialRes)  # Brunt-vaisala frequency squared???
                        bvMean = np.mean( bvF2[ np.nonzero( [ sum(x) for x in regions.T ] ) ] )  # Mean of bvF2 across cols of region
                        if not np.sqrt(bvMean) > intrinsicF > coriolisF:
                            print("Shit. And I was so close, too.")
                        else:
                            # Vertical wavelength [1/m] ?
                            m = 2 * np.pi / np.mean( (1.03 * scales)[ np.nonzero( [ sum(x) for x in regions.T ] ) ] )
                            # Horizontal wavelength [1/m] ?
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
                            altitudeOfDetection = np.mean(data['Alt'][ np.nonzero( [ sum(x) for x in regions.T ] ) ])
                            # Get index of mean altitude
                            detectionIndex = np.where(np.min(np.abs(data['Alt'] - altitudeOfDetection)))
                            # Get latitude at index
                            latitudeOfDetection = data['Lat.'][detectionIndex]
                            # Get longitude at index
                            longitudeOfDetection = data['Long.'][detectionIndex]
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

                            print("Wave Properties: ")
                            print(waveProp)

            plt.figure()
            plt.contourf(data['Alt'] / 1000, 1 / freq, power)
            cb = plt.colorbar()
            plt.contour(data['Alt'] / 1000, 1 / freq, plotter, colors='red')
            # plt.contour(data['Alt'], freq, power, colors='red')
            plt.xlabel("Altitude [km]")
            plt.ylabel("Period [m]")
            plt.title("Ummmmm")
            cb.set_label("Power [m^2/s^2]")

            if saveData:
                plt.savefig(savePath+"/"+file[0:-4]+"_power_surface.png")
            if showPlots:
                plt.show()
            plt.close()


            ########## FINISHED ANALYSIS ##########

            print("Finished analysis.")

print("\nAnalyzed all .txt files in folder /"+dataSource+"/")

