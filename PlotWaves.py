"""

This file contains code to plot detected gravity waves based on the JSON output files from DetectWaves.py

It accepts user input and plots all of the flights in the given directory as lines, placing the detected waves
as arrows on the plot. It does this task in either 2D or 3D, however, there seems to be a bug involved when
making a 3D plot where it's appropriate to use units of hours that I haven't yet figured out. The plotting works
well for making a 3D plot of the entire summer, or when making 2D plots, but I haven't yet created good plots
using sequential flights in 3D. Much more work is needed to make a polished product, but this should provide a
good start for plots of this nature. Currently, the program plots the waves in their propagation direction,
scaled by their intrinsic group velocity. However, this could change, and could conceivably be picked from a
list of options by the user during the user input section.

"""




# Make the necessary imports
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.dates as mdates
import datetime
import matplotlib.lines as mlines
from numpy.core.defchararray import lower
from WaveDetectionFunctions import getUserInputFile, getUserInputTF


# Get one of a number of options of units, for use with plotting
def getUserInputUnits():
    print("Enter the unit of time to use with this plot:")
    userInput = ""
    while not userInput:
        userInput = input()
        if lower(userInput) != "hours" and lower(userInput) != "days" and lower(userInput) != "months" and lower(userInput) != "years":
            print("Please enter either 'hours', 'days', 'months', or 'years':")
            userInput = ""

    return lower(userInput)

# Get the data source, units, 2d/3d, and title of plot from user
def getAllUserInput():
    dataSource = getUserInputFile("Enter path to data input directory:")
    unitHours = getUserInputUnits()
    plot3D = getUserInputTF("Would you like to generate a 3D plot?")
    print("Enter plot title:")
    plotTitle = input()

    # Build a dictionary to return values
    results = {
        'dataSource': dataSource,
        'units': unitHours,
        'plot3D': plot3D,
        'title': plotTitle
    }

    return results



# First, get the user input
userInput = getAllUserInput()

# Start a figure to plot on
fig = plt.figure()

# Either start a 3d axis, or a 2d axis
if userInput.get('plot3D'):
    ax = fig.gca(projection='3d')
else:
    ax = fig.gca()

# Set the x-axis to use dates instead of numbers
ax.xaxis_date()

# Set the correct formatter on the x-axis, according to the units
if userInput.get('units') == "hours":
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
elif userInput.get('units') == "days":
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
elif userInput.get('units') == "months":
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
elif userInput.get('units') == "years":
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# For every JSON file, do the following:
for file in os.listdir( userInput.get('dataSource') ):
    if not file.endswith(".json"):
        continue

    print("Reading file "+str(file))

    waves = {}
    # Read in the wave data as a python dictionary
    try:
        with open(os.path.join(userInput.get('dataSource'), file)) as json_file:
            data = json.load(json_file)
            waves = data.get('waves')
            flightPath = data.get('flightPath')
    except:
        print("JSON file does not contain wave data")
        continue

    # Define empty arrays, to be filled with values
    X = []
    Y = []
    U = []
    V = []
    W = []

    for wave in waves.values():
        # For every wave, get X and Y coordinates
        X.append(wave.get('Date and Time [UTC]'))

        Y.append(wave.get('Altitude [km]'))

        angle = wave.get('Propagation direction [deg]')
        mag = wave.get('Intrinsic horizontal group velocity [m/s]')
        # Set components of the wave
        U.append( mag * np.sin( angle * np.pi / 180 ) )
        V.append( mag * np.cos( angle * np.pi / 180 ) )
        W.append(wave.get('Intrinsic vertical group velocity [m/s]'))

    # Get the datetime objects from strings in the dictionary
    X = [datetime.datetime.strptime(date.split('.', 1)[0], '%Y-%m-%d %H:%M:%S') for date in X]

    # If 3d, pass 3d arguments to the quiver command
    if userInput.get('plot3D'):
        X = mdates.date2num(X)
        #x, y, z = np.meshgrid(X, np.zeros(len(X)), Y)

        #u, v, w = np.meshgrid(U, V, W)
        ax.quiver(X, np.zeros(len(X)), Y, U, V, W, color='red')

    else:
        # Otherwise, plot in 2d
        plt.quiver(X, Y, U, V, color='red')

    # Now, fix the format of the X and Y lists
    X = flightPath.get('time')
    X = [datetime.datetime.strptime(date[0].split('.', 1)[0], '%Y-%m-%d %H:%M:%S') for date in X]

    Y = flightPath.get('alt')
    Y = np.array(Y).reshape(len(X)) / 1000  # convert to km

    # And either plot the flight paths in 3d or 2d
    if userInput.get('plot3D'):
        X = mdates.date2num(X)
        ax.plot(X, np.zeros(len(X)), Y, color='blue')
    else:
        plt.plot( X, Y, color='blue')

# Define custom legend entries corresponding to the colors/shapes in the plot
blue_line = mlines.Line2D([], [], color='blue', label='Radiosonde flight')
red_arrow = mlines.Line2D([], [], color='w', marker=r'$\rightarrow$', markeredgecolor='red', markerfacecolor='red', markersize=15, label='Gravity Wave')
plt.legend(handles=[blue_line, red_arrow])

# Set the title, as well as axis labels
plt.title(userInput.get('title'))
if userInput.get('units') == "hours":
    plt.xlabel("Time [UTC]")
else:
    plt.xlabel("Date [UTC]")

if userInput.get('plot3D'):
    ax.set_zlabel("Altitude [km]")
    plt.ylim(-10, 10)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
else:
    plt.ylabel("Altitude [km]")

# Done, show the plot
plt.show()


