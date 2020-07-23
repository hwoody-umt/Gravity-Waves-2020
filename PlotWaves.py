import os
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.dates as mdates
import datetime
import matplotlib.lines as mlines
from numpy.core.defchararray import lower
from WaveDetectionFunctions import getUserInputFile

def getUserInputHD():
    print("Enter the unit of time to use with this plot:")
    userInput = ""
    while not userInput:
        userInput = input()
        if lower(userInput) != "hours" and lower(userInput) != "days" and lower(userInput) != "months" and lower(userInput) != "years":
            print("Please enter either 'hours', 'days', 'months', or 'years':")
            userInput = ""

    return lower(userInput)

def getAllUserInput():
    dataSource = getUserInputFile("Enter path to data input directory:")
    unitHours = getUserInputHD()
    print("Enter plot title:")
    plotTitle = input()

    # Build a dictionary to return values
    results = {
        'dataSource': dataSource,
        'units': unitHours,
        'title': plotTitle
    }

    return results


########## Real code (not functions) goes here ##########

userInput = getAllUserInput()
plt.gca().xaxis_date()
if userInput.get('units') == "hours":
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
elif userInput.get('units') == "days":
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
elif userInput.get('units') == "months":
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
elif userInput.get('units') == "years":
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

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

plt.title(userInput.get('title'))
plt.xlabel("Time [UTC]")
plt.ylabel("Altitude [km]")
plt.show()


