import numpy as np  # Numbers (like pi) and math
import os  # File reading and input
import WaveDetectionFunctions as fun


########## Function definitions, to be used later ##########


dataSource = fun.getUserInputFile("Enter path to data input directory: ")
saveData = fun.getUserInputTF("Would you like to save PBL output to the GRAWMET profiles?")


for file in os.listdir(dataSource):

    # Read the file to gather data
    data = fun.cleanData(file, dataSource)

    if data.empty:  # File is not a GRAWMET profile
        print("File does not appear to be a GRAWMET profile, quitting analysis...")
        continue  # Skip to next file and try again

    # Get launchDateTime for interpolateData() function, ignore default pblHeight
    launchDateTime, pblHeight = fun.readFromData(file, dataSource)
    del pblHeight

    spatialResolution = 1  # meters in between uniformly distributed data points, must be pos integer
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
    pblHeightRI = fun.pblRI(vpt, u, v, hi)  # RI method
    pblHeightVPT = fun.pblVPT(vpt, hi)  # Virtual potential temperature method
    pblHeightPT = fun.pblPT(hi, pot)  # Potential temperature method
    pblHeightSH = fun.pblSH(hi, rvv)  # Specific humidity method

    # Find min, max, and median PBL values
    pbls = [pblHeightSH, pblHeightVPT, pblHeightPT, pblHeightRI]
    pblHeightMax = np.max(pbls)
    pblHeightMin = np.min(pbls)
    pblHeightMedian = np.median(pbls)  # Median is the best

    print("Calculated PBL height (min, median, max) of (" + str(pblHeightMin) + ", " + str(pblHeightMedian) + ", " + str(pblHeightMax) + ") meters above ground level.")
    print(fun.layerStability(hi, pot))  # Print the layer stability, while we're at it

    ##### Now write max PBL height to profile file

    if saveData:

        stringToWrite = "Max PBL height:\t" + str(pblHeightMax) + "\t"
        stringToWrite += "Min PBL height:\t" + str(pblHeightMin) + "\t"
        stringToWrite += "Median PBL height:\t" + str(pblHeightMedian) + "\n\n"

        contents = []  # Initialize empty list

        with open(os.path.join(dataSource, file), "r") as f:
            contents = f.readlines()  # Read in entire file as list of lines

        beginIndex = [i for i in range(len(contents)) if contents[i].rstrip() == "Profile Data:"]  # Find beginning of main section
        beginIndex = beginIndex[0]

        # Insert pbl information into contents before main section begins
        #contents.insert(beginIndex, stringToWrite)
        contents.insert(beginIndex, "PBL Information:\n")

        with open(os.path.join(dataSource, file), "w") as f:
            f.writelines(contents)  # Write new contents to the original profile file
