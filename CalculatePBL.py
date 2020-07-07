import numpy as np  # Numbers (like pi) and math
import os  # File reading and input
import WaveDetectionFunctions as fun


########## Function definitions, to be used later ##########


dataSource = fun.getUserInputFile("Enter path to data input directory: ")


for file in os.listdir(dataSource):

    # Read the file to gather data
    data = fun.cleanData(file, dataSource)

    if data.empty:  # File is not a GRAWMET profile
        continue  # Skip to next file and try again

    # Get launchDateTime for interpolateData() function, ignore default pblHeight
    launchDateTime, pblHeight = fun.readFromData(file, dataSource)
    del pblHeight

    spatialResolution = 1  # meters in between uniformly distributed data points, must be pos integer
    # Interpolate to clean up the data, fill holes, and make a uniform spatial distribution of data points
    data = fun.interpolateData( data, spatialResolution, 0, launchDateTime )  # Start at zero meters to find PBL

    if data.empty:  # File has too much sequential missing data, analysis will be invalid
        continue  # Skip to next file and try again

    print("Calculating PBL height for "+str(file))

    ##### The following are all the variables and equations needed for predicting PBL Height

    hi = data['Alt'] - data['Alt'][0]  # height above ground in meters
    epsilon = 0.622  # epsilon, unitless constant

    # vapor pressure
    e = 6.1121 * np.exp((18.678 - (data['T'] / 234.84)) * (data['T'] / (257.14 + data['T']))) * data[
        'Hu']  # hPa

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

    # CALL THE FUNCTIONS YOU WANT TO USE
    # Only functions that are called in this section will display in the output data
    pblHeightRI = fun.pblRI(vpt, u, v, hi)  # RI method
    pblHeightVPT = fun.pblVPT(pot, rvv, vpt, hi)  # Virtual potential temperature method
    pblHeightPT = fun.pblPT(hi, pot)  # Potential temperature method
    pblHeightSH = fun.pblSH(hi, rvv)  # Specific humidity method

    pbls = [pblHeightSH, pblHeightVPT, pblHeightPT, pblHeightRI]
    pblHeightMax = np.max(pbls)
    pblHeightMin = np.min(pbls)
    pblHeightMean = np.mean(pbls)

    print("Calculated PBL (min, mean, max) heights of (" + str(pblHeightMin) + ", " + str(pblHeightMean) + ", " + str(pblHeightMax) + ")")
    print(fun.layerStability(hi, pot))  # Print the layer stability, while we're at it

    ##### Now write max PBL height to profile file

    stringToWrite = "Max PBL height: " + str(pblHeightMax) + " "
    stringToWrite += "Min PBL height: " + str(pblHeightMin) + " "
    stringToWrite += "Mean PBL height: " + str(pblHeightMean)

    with open(os.path.join(dataSource, file), "rw") as f:
        contents = f.readlines()  # Read in entire file as list of lines

        beginIndex = (contents.rstrip() == "Profile Data:")  # Find beginning of main section

        # Insert pbl information into contents before main section begins
        contents.insert(beginIndex, "PBL Information:")
        contents.insert(beginIndex, stringToWrite)

        f.writelines(contents)  # Write new contents to the original profile file
