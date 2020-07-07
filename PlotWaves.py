import os
import matplotlib.pyplot as plt
import numpy as np
import json
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import matplotlib.dates as mdates
import datetime
import pandas


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
    showPlots = getUserInputTF("Do you want to display the output plots?")
    saveData = getUserInputTF("Do you want to save the output plots?")
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


########## Real code (not functions) goes here ##########

userInput = getAllUserInput()
plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

print(os.listdir( userInput.get('dataSource') ))

for file in os.listdir( userInput.get('dataSource') ):
    if not file.endswith(".json"):
        continue

    print("Reading file "+str(file))

    waves = {}

    try:
        with open(os.path.join(userInput.get('dataSource'), file)) as json_file:
            data = json.load(json_file)
            waves = data.get('waves')
    except:
        print("JSON file does not contain wave data")
        continue

    X = []
    Y = []
    U = []
    V = []

    for wave in waves.values():

        X.append(wave.get('Date and Time [UTC]'))

        Y.append(wave.get('Altitude [km]'))

        angle = wave.get('Angle of wave [deg]')
        mag = wave.get('Intrinsic horizontal group velocity [m/s]')

        U.append( mag * np.sin( angle * np.pi / 180 ) )
        V.append( mag * np.cos( angle * np.pi / 180 ) )

    X = [datetime.datetime.strptime(date.split('.', 1)[0], '%Y-%m-%d %H:%M:%S') for date in X]

    plt.quiver(X, Y, U, V, color='red')
    Y = [Y for (X, Y) in sorted(zip(X, Y)) ]
    X = sorted(X)
    plt.plot( X, Y, color='blue')
plt.xlabel("Date and Time [UTC]")
plt.ylabel("Altitude [km]")
plt.show()


