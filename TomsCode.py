import numpy as np  # Numbers (like pi) and math
import matplotlib.pyplot as plt  # Easy plotting
import pandas as pd  # Convenient data formatting, and who doesn't want pandas
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import os  # File reading and input
from io import StringIO  # Used to run strings through input/output functions
import pywt  # Library PyWavelets, for wavelet transforms

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
    savePath = "NA"
# MATLAB code has lower and upper altitude cut-offs and latitude
# I've changed these to be read in from the data

# For debugging, print results
print("Running with the following parameters:")
print("Path to input data: /"+dataSource+"/")
print("Display plots: "+str(showPlots))
print("Save data: "+str(saveData))
print("Path to output data: "+savePath+"\n")

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
                else:
                    for col in range(data.shape[1]):
                        if data.iloc[row, col] == 999999.0:  # This value appears a lot and is obviously wrong
                            badRows.append(row)
                            break
            if len(badRows) > 0:
                print("Dropping "+str(len(badRows))+" rows containing unusable data")
            data = data.drop(data.index[badRows])

            #Make cursory inspection plots, dependent on user input showPlots
            if showPlots:
                print("Displaying input data")
                plt.plot(data['Long.'], data['Lat.'])
                plt.ylabel("Latitude")
                plt.xlabel("Longitude")
                plt.show()

            ########## PERFORMING ANALYSIS ##########

            #Calculate variables needed for further analysis

            tk = data['T'] + 273.15  # Temperature in Kelvin
            hi = data['Alt'] - data['Alt'][1]  # height above ground in meters
            epsilon = 0.622  # epsilon, unitless constant

            # saturation vapor pressure
            es = 6.1121 * np.exp((18.678 - (data['T'] / 234.84)) * (data['T'] / (257.14 + data['T'])))  # hPa

            # vapor pressure
            e = es * data['Hu']  # hPa

            # water vapor mixing ratio
            rvv = (epsilon * e) / (data['P'] - e)  # unitless

            # potential temperature
            pot = (1000.0 ** 0.286) * tk / (data['P'] ^ 0.286)  # kelvin

            # virtual potential temperature
            vpt = pot * ((1 + (rvv / epsilon)) / (1 + rvv))  # kelvin

            # u and v (east & north?) components of wind speed
            u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
            v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

            g = 9.81  # m/s/s




            def PBLri(vpt,Alt,g,u,v):
                # This function calculates richardson number. We then
                # search for where Ri(z) is near 0.25 and interpolates to get the height
                # z where Ri(z) = 0.25.
                #
                # INPUTS: temperature in C, temperature in K, altitude in meters, height in
                # meters, pressure in hPa, humidity as decimal value, wind speed in m/s,
                # and wind direction in degrees.
                #
                # OUTPUTS: richardson number
                ri = (((vpt - vpt[0]) / vpt[0]) * (Alt - Alt[0]) * g) / (u ** 2 + v ** 2)

                # Richardson number. If surface wind speeds are zero, the first data point
                # will be an inf or NAN.
                return ri



            # Next, figure out what the preprocessing is actually accomplishing and why.
            # It seems to be creating a new data set by picking several times and then
            # doing a linear interpolation between them? Why?

            # Then, work on the coriolis frequency... dependent on latitude, but
            # also assumed to be constant? Use mean latitude? Or treat as variable?

            # The wavelet transform code involves multiple imported methods, so
            # I need to look the PyWavelet library and really
            # understand the math behind the wavelet transform in order to adapt
            # the code to python.

            ########## FINISHED ANALYSIS ##########

            print("Finished analysis.")

print("\nAnalyzed all .txt files in folder /"+dataSource+"/")

