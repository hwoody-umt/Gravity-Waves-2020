from io import StringIO
import numpy as np  # Numbers (like pi) and math
import os  # File reading and input
import WaveDetectionFunctions as fun
import pandas as pd


########## Function definitions, to be used later ##########

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
        # noinspection PyChainedComparisons
        if not str(data['Rs'].loc[row]).replace('.', '', 1).isdigit():  # Check for nonnumeric or negative rise rate
            badRows.append(row)
        # Check for stable or decreasing altitude (removes rise rate = 0)
        elif row > 0 and np.diff(data['Alt'])[row-1] <= 0:
            badRows.append(row)
        else:
            for col in range(data.shape[1]):  # Iterate through every cell in row
                if data.iloc[row, col] == 999999.0:  # This value appears to be GRAWMET's version of NA
                    badRows.append(row)  # Remove row if 999999.0 is found
                    break

    if len(badRows) > 0:
        print("Dropping "+str(len(badRows))+" rows containing unusable data")
        data = data.drop(data.index[badRows])  # Actually remove any necessary rows
    data.reset_index(drop=True, inplace=True)  # Return data frame index to [0,1,2,...,nrow]

    # Get rid of extraneous columns that won't be used for further analysis
    essentialData = ['Time', 'Alt', 'T', 'P', 'Ws', 'Wd', 'Dewp.']
    data = data[essentialData]

    return data  # return cleaned pandas data frame

########## Actual Code ##########

dataSource = fun.getUserInputFile("Enter path to data input directory: ")
saveData = fun.getUserInputTF("Would you like to save PBL output to the GRAWMET profiles?")


for file in os.listdir(dataSource):

    # Read the file to gather data
    data = cleanData(file, dataSource)

    if data.empty:  # File is not a GRAWMET profile
        print("File does not appear to be a GRAWMET profile, quitting analysis...\n")
        continue  # Skip to next file and try again

    # Get launchDateTime for interpolateData() function
    launchDateTime, pblHeight = fun.readFromData(file, dataSource)
    # If PBL height isn't the default 1500 meters, then file has previously been analyzed, so skip
    if pblHeight != 1500:
        continue

    spatialResolution = 5  # meters in between uniformly distributed data points, must be pos integer
    # Interpolate to clean up the data, fill holes, and make a uniform spatial distribution of data points
    data = fun.interpolateData( data, spatialResolution, 0, launchDateTime )  # Start at zero meters to find PBL

    if data.empty:  # File has too much sequential missing data, analysis will be invalid
        continue  # Skip to next file and try again

    print("Calculating PBL height")

    ##### The following are all the variables and equations needed for predicting PBL Height

    hi = data['Alt'] - data['Alt'][0]  # height above ground in meters

    epsilon = 0.622  # epsilon, unitless constant

    # vapor pressure
    e = np.exp(1.8096 + (17.269425 * data['Dewp.']) / (237.3 + data['Dewp.']))  # hPa

    # water vapor mixing ratio
    rvv = np.divide( np.multiply( np.array(e), epsilon ), ( np.array(data['P']) - np.array(e) ) )  # unitless

    # potential temperature
    pot = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286)  # kelvin

    # virtual potential temperature
    #vpt = pot * ((1 + (rvv / epsilon)) / (1 + rvv))  # kelvin
    virtcon = 0.61
    vpt = pot * (1 + (virtcon * rvv))

    # absolute virtual temperature
    #vt = (data['T'] + 273.15) * ((1 + (rvv / epsilon)) / (1 + rvv))  # kelvin

    # u and v (east & north?) components of wind speed
    u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
    v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

    # CALL THE FUNCTIONS YOU WANT TO USE
    # Only functions that are called in this section will display in the output data
    pblHeightRI = float(pblRI(vpt, u, v, hi))  # RI method
    pblHeightVPT = float(pblVPT(vpt, hi))  # Virtual potential temperature method
    pblHeightPT = float(pblPT(hi, pot))  # Potential temperature method
    pblHeightSH = float(pblSH(hi, rvv))  # Specific humidity method

    # Find min, max, and median PBL values
    pbls = [pblHeightSH, pblHeightVPT, pblHeightPT, pblHeightRI]
    pblHeightMax = np.max(pbls)
    pblHeightMin = np.min(pbls)
    pblHeightMedian = np.median(pbls)  # Median is the best

    print("Calculated PBL height (min, median, max) of (" + str(pblHeightMin) + ", " + str(pblHeightMedian) + ", " + str(pblHeightMax) + ") meters above ground level.")

    ##### Now write max PBL height to profile file

    if saveData:
        stringToWrite = "Min PBL height:\t" + str(pblHeightMin) + "\t"
        stringToWrite += "Median PBL height:\t" + str(pblHeightMedian) + "\t"
        stringToWrite += "Max PBL height:\t" + str(pblHeightMax) + "\n\n"


        contents = []  # Initialize empty list

        with open(os.path.join(dataSource, file), "r") as f:
            contents = f.readlines()  # Read in entire file as list of lines

        beginIndex = [i for i in range(len(contents)) if contents[i].rstrip() == "Profile Data:"]  # Find beginning of main section
        beginIndex = beginIndex[0]

        # Insert pbl information into contents before main section begins
        contents.insert(beginIndex, stringToWrite)
        contents.insert(beginIndex, "PBL Information:\n")

        with open(os.path.join(dataSource, file), "w") as f:
            f.writelines(contents)  # Write new contents to the original profile file
