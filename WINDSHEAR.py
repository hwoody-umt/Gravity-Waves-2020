#!/usr/bin/env python
# coding: utf-8

# In[39]:



#get data from profile data file
##GET DATA##

import numpy as np  # Numbers (like pi) and math
import matplotlib.pyplot as plt  # Easy plotting
import pandas as pd  # Convenient data formatting, and who doesn't want pandas
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import os  # File reading and input
from io import StringIO  # Used to run strings through input/output functions
from scipy import interpolate  # Used for PBL calculations
import pywt  # Library PyWavelets, for wavelet transforms

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

            
####### FUNCTIONS ################

#cut out data from after balloon burst

def maxaidx (Ws, Alt):
    ws = data['Ws'][1:max(data['Ws'])]
    height = data['Alt'][1:max(data['Alt'])]-data['Alt(1)']
    return ws, height

#find local maximums

#using Ri method to find areas of instability
#this method will look for ri levels between 0 and 0.25, known parameters for KH instability among layers of atmosphere

import math

tk = data['T'] +273.15 #Temperature in Kelvin
hi = data['Alt'] - data['Alt'][1] #height above ground in meters

#epsilon, unitless constant
epsilon = 0.622 

#saturation vapor pressure
es = 6.1121 * np.exp((18.678 - (data['T'] / 234.84)) * (data['T'] / (257.14 + data['T']))) * data['Hu']

#vapor pressure
e = es*data['Hu'] #hPa

#water vapor mixing ratio
rvv = (epsilon*e)/(data['P']-e) #unitless

#potential temperature
pot = (1000.0 ** 0.286) * (data['T'] + 273.15) / (data['P'] ** 0.286) #kelvin

#virtual potential temperature
vpt = pot*((1+(rvv/epsilon))/(1+rvv)) #kelvin

#component wid speeds, m/s
u = -data['Ws'] * np.sin(data['Wd'] * np.pi / 180)
v = -data['Ws'] * np.cos(data['Wd'] * np.pi / 180)

hi0 = data['Alt'][1]; #surface altitude
vpt0 = vpt[1] #virtual potential temperature at surface
g = 9.81

#Richardson number. If surface wind speeds are zero, the first data point
#will be an inf or NAN.

#intialize lists
significant_ri = []
sig_ri_final = []
height_sigri =[] 
res =[]


#find ri levels below 0.25 and above 0
#then finds the height of these significant ri levels and returns of list of heights with suspect ri

def ri_level(T, tk, hi, P, Hu, ws, Wd):
    g = 9.81
    ri = (pot - pot[0]) * hi * g / ( pot * (u ** 2 + v ** 2) )
    for value in ri:
        if value <= 0.25:
            significant_ri.append(value)
    for sigval in significant_ri:
        if sigval >= 0:
            sig_ri_final.append(sigval)
    res = [key for key, val in enumerate(sig_ri_final)
           if val in set(hi)]
    return res

#wind shear every 1000m

import numpy as np
from scipy.signal import argrelextrema

ws = data['Ws']

lmx = argrelextrema(ws.values, np.greater)
lmn = argrelextrema(ws.values, np.less)

#intialize empty list
Levelsx = []

#def windshear(lmx, hi):
    #for k in lmx: #looks throught the local maximums to evaluate each one
        #area = lmx.find(hi >= hi(k) + 1000) #defines a layer between the local max and 1000m above the max
        #if area == 0: 
           # print ("Larger than Burst Height")
        #wl = argrelextrema(ws[k:area], np.less) #finds the local minimum in the above defined layer of 1000m
        #if ws(k) - wl >= 7.5: #looks for a windspeed change greater than or equal to 7.5 m/s
            #print ("Significant windshear at " + hi(k) + "meters")
            #levelsx.append(hi(k)) #creates list of heights with significant windshear

            
            
            
            #######PERFORMING ANALYSIS#########
                
print("Performing Analysis")
windshearRI = ri_level(data['T'], tk, hi, data['P'], data['Hu'], ws, data['Wd'])
#windshearSL = windshear(lmx, hi)
print("Calculated Windshears with RI method " + str(windshearRI)) #+ " Calculated Windshears using Jaxen's method " +str(windshearSL))
                
                
                
                
print("Finished analysis.")

print("\nAnalyzed all .txt files in folder /"+dataSource+"/")


# In[ ]:




