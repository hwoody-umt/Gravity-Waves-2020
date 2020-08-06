## Gravity-Waves-2020


 *Developed by Keaton Blair and Hannah Woody, MSGC*


 This folder contains python code to detect and analyze gravity
 waves from radiosonde data, as well as sample data and output.


### Descriptions of the python files are as follows:


 CalculatePBL.py:
 Run this file to read a GRAWMET profile, estimate the height of the PBL,
 and write that PBL height into the profile so that it can be used by
 the file DetectWaves.py, which either reads PBL height from the profile
 or defaults to 1500 meters. The file takes user input, asking for the
 path to a directory containing profiles, and whether or not to save the
 data to those profiles.


 DetectWaves.py:
 Run this file to analyze GRAWMET profile data. The file takes in user input,
 asking for the path to a directory containing the profile data, whether or
 not to display power surfaces, and whether or not to save the images and
 analysis files to a user provided directory. While running, this code also
 displays plots comparing two different hodographs and asks users to choose
 the more elliptical shape, and type the name of the method that yielded it
 (shown above the plot) into the console.


 PlotWaves.py:
 Run this file to make specific plots of the output files from DetectWaves.py.
 This file takes in user input, asking for the path to a directory containing
 the output files, the units of time to plot on the x-axis, whether to make a
 2D or 3D plot, and what the title of the plot should be.


 TorrenceCompoWavelets.py:
 See description contained in file. https://github.com/chris-torrence/wavelets/tree/master/wave_python
 Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
            Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.


 WaveDetectionFunctions.py:
 This file contains all of the functions necessary to run DetectWaves.py. Keep
 this file in the same directory, or change the path in the import statement at
 the top of DetectWaves.py to match the location of this file.


### Descriptions of the folders are as follows:


 Summer_2020_Flights:
 This contains GRAWMET profile data collected from radiosonde flights by UM
 BOREALIS, sponsored by MSGC and NSF, during the summer of 2020.


 Summer_2020_Output:
 This contains output resulting from running DetectWaves.py on the data in
 Summer_2020_Flights, as well as a plot created in part using PlotWaves.py.