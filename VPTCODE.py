#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np  # Numbers (like pi) and math
import matplotlib.pyplot as plt  # Easy plotting
import pandas as pd  # Convenient data formatting, and who doesn't want pandas
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import os  # File reading and input
from io import StringIO  # Used to run strings through input/output functions
from scipy import interpolate  # Used for PBL calculations
import pywt  # Library PyWavelets, for wavelet transforms

########## Function definitions, to be used later ##########

def pblri(vpt, vt, pt, u, v, Alt):
    # This function calculates richardson number. It then
    # searches for where Ri(z) is near 0.25 and interpolates to get the height
    # z where Ri(z) = 0.25.
    #
    # INPUTS: write what these are eventually
    #
    # OUTPUTS: PBL height based on RI

    g = 9.81  # m/s/s
    ri = (((pt - pt[0]) * Alt * g) / ( pt * (u ** 2 + v ** 2) ))
    # This equation is right according to
    #https://www.researchgate.net/figure/Profile-of-potential-temperature-MR-and-Richardson-number-calculated-from-radiosonde_fig4_283187927
    #https://resy5.iket.kit.edu/RODOS/Documents/Public/CD1/Wg2_CD1_General/WG2_RP97_19.pdf

    #vt = vt[0:len(vt)-1]
    #ri = (np.diff(vpt) * np.diff(hi) * g / abs(vt)) / (np.diff(u) ** 2 + np.diff(v) ** 2)
    #print(ri)
    # Richardson number. If surface wind speeds are zero, the first data point
    # will be an inf or NAN.

    # Interpolate between data points
    riCutOff = 0.33
    f = interpolate.UnivariateSpline(hi, ri - riCutOff, s=0)
    plt.plot(ri, hi)
    plt.plot(f(hi)+ riCutOff, ri)
    plt.plot([0.33] * 2, plt.ylim())
    plt.xlabel("RI")
    plt.ylabel("Height above ground [m]")
    plt.axis([-10, 20, 0, 5000])
    plt.show()

    # Return heights where interpolation crosses riCutOff = 0.25
    # Need a way to pick which one is the right one... there are many
    if len(f.roots()) == 0:
        return [0]
    return max(f.roots())


def pbl_ri_test(vpt, vt, pt, u, v, hi):
    g = 9.81  # m/s/s
    ri = (((pt - pt[0]) * hi * g) / ( pt * (u ** 2 + v ** 2) ))
    ri_list_one = []
    ri_list_two = []
    for value, alt_value in zip(ri, hi):
        if value <= 0.25 and value >= 0:
            ri_list_one.append(value)
            ri_list_two.append(alt_value)
    return ri_list_two


def pblpt(hi, pot):
    maxhidx = np.argmax(hi)
    pth = pot[10:maxhidx]
    upH = hi[10:maxhidx]
    topH = 3500
    height3k = []
    for i, H in enumerate(upH):
        if H >= topH:
            break
        height3k.append(H)
    pt3k = pth[0:i]
    dp = np.gradient(pt3k, height3k)
    maxpidx = np.argmax(dp)
    pbl_potential_temperature = height3k[maxpidx] 
    
    import matplotlib.pyplot as plt
    plt.plot(dp, height3k)
    plt.show()
    
    return str(pbl_potential_temperature)



def pblsh(hi, rvv):
    maxhidx = np.argmax(hi)
    q = rvv/(1+rvv)
    qh = q[10:maxhidx]
    upH = hi[10:maxhidx]
    topH = 3500
    height3k = []
    for i, H in enumerate(upH):
        if H >= topH:
            break
        height3k.append(H)
    qh3k = qh[0:i]
    dp = np.gradient(qh3k, height3k)
    minpix = np.argmin(dp)
    pbl_sh = height3k[minpix]
    
    import matplotlib.pyplot as plt
    plt.plot(dp, height3k)
    plt.show()
    
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
    if len(rootdos) == 0:
        return [0.0]
    return max(rootdos)





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
    
def largestpbl (pblHeightRI, pblHeightVPT, pblHeightPT, pblHeightSH):
    listofheights = [str(pblHeightRI), str(pblHeightPT), str(pblHeightSH), str(pblHeightVPT)]
    Output = sorted(listofheights, key = lambda x:float(x))
    return Output[-1]

def drawPlots(alt, t, td, pblHeightRI, pblHeightVPT):#, pblHeightPT, pblHeightSH):
    print("Displaying data plots")

    # Plot radiosonde path
    plt.plot(data['Long.'], data['Lat.']) 
    plt.ylabel("Latitude [degrees]")
    plt.xlabel("Longitude [degrees]")
    plt.title("Radiosonde Flight Path")
    plt.show()

    # Plot pbl estimates
    pblHeightRI += alt[0]  # Convert height to altitude
    #pblHeightSH += alt[0]
    plt.plot(t, alt, label="Temperature")
    plt.plot(td, alt, label="Dewpoint")
    #plt.plot(plt.get_xlim(),[pblHeightPT] * 2, label="PT Method")
    plt.plot(plt.xlim(), [pblHeightRI] * 2, label="RI Method")
    plt.plot(plt.xlim(), [pblHeightVPT] * 2, label = "VPT Method")
    plt.axis([-80, 20, 1000, 3500])
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

            ########## PERFORMING ANALYSIS ##########
            
            #Calculate variables needed for further analysis
            print("Calculating PBL height")
            
            
            hi = data['Alt'] - data['Alt'][0]  # height above ground in meters
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
            
            vptn = vpt[0]

            # Get three different PBL height estimations
            pblHeightRI = pblri(vpt, vt, pot, u, v, hi)
            pblHeightRITRIAL = pbl_ri_test(vpt, vt, pot, u, v, hi)
            pblHeightVPT = pblvpt(pot, rvv, vpt, hi)
            pblHeightPT = pblpt(hi, pot)
            pblHeightSH = pblsh(hi, rvv)
            #pblLargestHeight = largestpbl(pblHeightRI, pblHeightVPT, pblHeightPT, pblHeightSH)
            print("Calculated PBL height of "+str(pblHeightRI))
            print:"Calculated PBL height of TRIAL METHOD " +str(pblHeightRITRIAL)
            print("Calculated PBL height PT METHOD of: " + str(pblHeightPT))
            print("Calculated PBL height SH METHOD of: " + str(pblHeightSH))
            print("Calculated PBL height VPT METHOD of: " + str(pblHeightVPT))
            print(layerStability(hi, pot))
            #print("HIGHEST PBL HEIGHT CALCULATED " + str(pblLargestHeight))

            # Make preliminary analysis plots, dependent on user input showPlots
            if showPlots:
                drawPlots(data['Alt'],data['T'],data['Dewp.'],pblHeightRI,pblHeightVPT)#,pblHeightPT,pblHeightSH)

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


# In[ ]:





# In[ ]:




