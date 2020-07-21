import os
import matplotlib.pyplot as plt
import numpy as np
import json
from numpy.core.defchararray import lower  # For some reason I had to import this separately
import matplotlib.dates as mdates
import datetime
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers


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

quiverLegendHandle = 0

for file in os.listdir( userInput.get('dataSource') ):
    if not file.endswith(".json"):
        continue

    print("Reading file "+str(file))

    waves = {}

    try:
        with open(os.path.join(userInput.get('dataSource'), file)) as json_file:
            data = json.load(json_file)
            waves = data.get('waves')
            flightPath = data.get('flightPath')
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

    X = flightPath.get('time')
    X = [datetime.datetime.strptime(date.split('.', 1)[0], '%Y-%m-%d %H:%M:%S') for date in X]

    Y = flightPath.get('alt')
    Y = np.array(Y) / 1000  # convert to km

    plt.plot( X, Y, color='blue')


blue_line = mlines.Line2D([], [], color='blue', label='Radiosonde flight')
red_arrow = mlines.Line2D([], [], color='w', marker=r'$\rightarrow$', markeredgecolor='red', markerfacecolor='red', markersize=15, label='Gravity Wave')
plt.legend(handles=[blue_line, red_arrow])

plt.title("Week 7 Radiosonde Flight")
plt.xlabel("Time [UTC]")
plt.ylabel("Altitude [km]")
plt.show()


