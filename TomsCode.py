import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.core.defchararray import lower
import os

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
savePath = getUserInputFile("Enter path to data output directory: ")
showPowerSurfaces = getUserInputTF("Do you want to show power surfaces?")
saveData = getUserInputTF("Do you want to save the output data?")
# MATLAB code has lower and upper altitude cut-offs and latitude
# I've changed these to be read in from the data

# For debugging, print results
print("Running with the following parameters:")
print("Path to input data: "+dataSource)
print("Path to output data: "+savePath)
print("Show power surfaces: "+str(showPowerSurfaces))
print("Save data: "+str(saveData)+"\n")

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
                print("File "+file+" contains GRAWMET profile data.")
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
            print("Writing data to temp file")
            f = open(os.path.join(savePath, ".temp.txt"), 'w')
            f.write(contents)
            f.close()
            del index
            del contents

            # Read in the data
            data = pd.read_csv(os.path.join(savePath, ".temp.txt"), delim_whitespace=True)

            ########## NEED CODE TO FIND PBL HERE ##########

            # Find the end of usable data
            badRows = []
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    if data.iloc[row, col] == 999999.0:  # This value seems to mark post-ascent data
                        badRows.append(row)
                        break
            print("Dropping "+str(len(badRows))+" rows containing unusable data")
            data = data.drop(data.index[badRows])

            ########## PERFORMING ANALYSIS ##########

            # Get u and v (east & north?) components of wind
            u = -data['Ws'] * np.sin(data['Wd']*np.pi/180)
            v = -data['Ws'] * np.cos(data['Wd']*np.pi/180)

            # Next, figure out what the preprocessing is actually accomplishing and why.

            # Then, work on the coriolis frequency... dependent on latitude, but
            # also assumed to be constant? Use mean latitude? Or treat as variable?

            # Finally, get to the wavelet transform and really fuck some shit up.

            ########## FINISHED ANALYSIS ##########

            os.remove(os.path.join(savePath, ".temp.txt"))
            print("Finished analysis.")

print("\nAnalyzed all .txt files in folder "+dataSource)
