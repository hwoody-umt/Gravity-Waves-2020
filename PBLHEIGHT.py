#!/usr/bin/env python
# coding: utf-8

# In[1]:


###########################################################################
###########################################################################
#           PBL Height Determination with 3 Different Methods             #        
###########################################################################
###########################################################################
#
#
# This code calculates planetary boundary height (PBL) using three 
# different methods. Each method outputs a height and the methods are 
# plotted with the PBLHs found. 
#
# The first method is the Richardson number method. The PBL height is
# defined to be the lowest altitude where richardson number Ri(z) = 0.25. 
# 
# The second methd is the potential temperature gradient method. The
# vertical gradient of potential temperature is calculated and the maximum
# of the gradient is defined as the PBL height.
#
# The third method is the specific humidity gradient method. The vertical
# gradient of specific humidity is calculated and the minimum of the
# gradient is defined as the PBL height. 
#
#
###########################################################################
###########################################################################
#                         Set up input prompt window                      #
###########################################################################
###########################################################################



# In[13]:


# This function calculates richardson number along with the variables (like
# potential temperature) needed to calculate richardson number. It then
# searches for where Ri(z) is near 0.25 and interpolates to get the height
# z where Ri(z) = 0.25.
#
# INPUTS: temperature in C, temperature in K, altitude in meters, height in
# meters, pressure in hPa, humidity as decimal value, wind speed in m/s,
# and wind direction in degrees.
#
# OUTPUTS: PBL height, potential temperature, water vaport mixing ratio, and
# richardson number 

import math

tk = T+273.15; #Temperature in Kelvin
hi = Alt-Alt(1); #height above ground in meters

#epsilon, unitless constant
epsilon = 0.622; 

#saturation vapor pressure
es = 6.1121*math.exp((18.678-(T/234.84))*(T/(257.14+T))); #hPa

#vapor pressure
e = es*Hu; #hPa

#water vapor mixing ratio
rvv = (epsilon*e)/(P-e); #unitless

#potential temperature
pot = (1000.0^0.286)*tk/(P^0.286); #kelvin

#virtual potential temperature
vpt = pot*((1+(rvv/epsilon))/(1+rvv)); #kelvin

#component wid speeds, m/s
u = Ws*cos(deg2rad(Wd)); 
v = Ws*sin(deg2rad(Wd));

Alt0 = Alt(1); #surface altitude
vpt0 = vpt(1); #virtual potential temperature at surface
g = 9.81;

#Richardson number. If surface wind speeds are zero, the first data point
#will be an inf or NAN.

def PBLri(T, tk, Alt, hi, P, Hu, Ws, Wd):
    ri = (((vpt-vpt0)/vpt0)*(Alt-Alt0)*g)/((u)^2+(v)^2)
    return ri


# In[ ]:





# In[12]:


idxri25G = []
idxri25L = []

idxri25G.append(ri.find(ri>=0.25)) #indices of ri values greater than 0.25
idxri25L.append(ri.find(ri<=0.25)) #indices of ri values less than 0.25

if len(idxri25L) == 0:
    pblri = 0 #if there is no ri value <0.25 then the pbl height is zero
else:
    if idxri25G(1)>idxri25L(1): #if ri starts below 0.25 then increases to above 0.25, idxri25G(1) is point right above 0.25
        upper = idxri25G(1)
        lower = idxri25G(1)-1
        pblri = np.interp(ri[lower:upper],hi[lower:upper],0.25)
    if idxri25G(1)<idxri25L(1): #if ri starts above 0.25 then decreases to below 0.25, idxri25L(1) is point right above 0.25
        upper = idxri25L(1) 
        lower = idxri25L(1)-1
        pblri = np.interp(ri[lower:upper],hi[lower:upper],0.25,)


# In[18]:


###########################################################################################################################
###########################################################################################################################
#                                               Potential Temp Method                                                     #
###########################################################################################################################
###########################################################################################################################


# In[8]:


def pblpt (hi, pot):
       maxhidx = max(hi)
       pth = pot[10:maxhidx]
       upH = hi[10:maxhidx]
       topH = 3500
       height3k = upH(upH<=topH)
       pt3k = pth(upH<=topH)
       dp3k = np.gradient(pt3k,height3k)
       dp = np.gradient(pot,hi)
       maxdpidx = max(dp3k)
       pblpt = height3k * maxdpidx
       return pblpt


# In[20]:


###########################################################################################################################
###########################################################################################################################
#                                               Specific Humidity Method                                                     #
###########################################################################################################################
###########################################################################################################################
    


# In[7]:


def pblsh (hi, rvv):
    maxhidx = max(hi)
    q = rvv/(1+rvv)
    qh = q[10:maxhidx]
    upH = hi[10:maxhidx]
    topH = 3500
    height3k = upH(upH<=topH)
    q3k = qh(upH<=topH)
    dq3k = np.gradient(q3k,height3k)
    dq = np.gradient(q,hi)
    mindpidx = min(dq3k)
    pblsh = height3k * mindpidx
    return pblsh


# In[22]:


###########################################################################################################################
###########################################################################################################################
#                                               Atmospheric Stability                                                     #
###########################################################################################################################
###########################################################################################################################


# In[1]:


def layer_stability (hi, pot):
    ds = 1
    du = 0.5
    m150 = hi.find(hi >= 150)
    start = m150(1)
    diff = pot(start)-pot(1)

    if diff < -ds:
        return print("Convective Boundary Layer")
    elif diff > ds:
        return print("Stable Boundary Layer")
    else:
        return print("Neutral Residual Layer")


# In[14]:


###########################################################################################################################
###########################################################################################################################
#                                                       Plotting                                                          #
###########################################################################################################################
###########################################################################################################################


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


import numpy as np

if ch1_pt:
    if ax_1 == 0:
        f1 = figure()
        ax_1 = axes(f1)
        
    cla(ax_1)
    plt.plot(ax_1, dpt, height)
    plt.hold(True)
    plt.plot(ax_1, PBLpt, 'g', label = "\theta Method")
    plt.plot(ax_1, PBLri, 'r', label = "Ri Method")
    plt.plot(ax_1, PBLsh, 'b', label = "Specific Hu Method")
    plt.hold(False)
    plt.title('Vertical Gradient of Potential Temperature')
    plt.axis(ax_1, [-0.1,0.1,0,4000])
    plt.show


# In[17]:


if ch2_ri:
    if ax_2 == 0:
        f2 = figure()
        ax_2 = axes(f2)
        
    cla(ax_2)
    plt.plot(ax_2, ri, height)
    plt.hold(True)
    plt.plot(ax_2, PBLpt, 'g', label = "\theta Method")
    plt.plot(ax_2, PBLri, 'r', label = "Ri Method")
    plt.plot(ax_2, PBLsh, 'b', label = "Specific Hu Method")
    plt.plot(ax_2, 0.25, label = "Ri(z) = 0.25")
    plt.hold(False)
    plt.title('Height vs. Richardson Number')
    plt.axis(ax_2, [-3,3,0,4000])
    plt.show


# In[19]:


if ch3_sh:
    if ax_3 == 0:
        f3 = figure()
        ax_3 = axes(f3)
        
    cla(ax_3)
    plt.plot(ax_3, dsh, height)
    plt.hold(True)
    plt.plot(ax_3, PBLpt, 'g', label = "\theta Method")
    plt.plot(ax_3, PBLri, 'r', label = "Ri Method")
    plt.plot(ax_3, PBLsh, 'b', label = "Specific Hu Method")
    plt.hold(False)
    plt.title('Vertical Gradient of Specific Humidity')
    plt.axis(ax_3, [-0.000035,0.000035,0,4000])
    plt.show


# In[20]:


if ax_4 == 0:
    f4 = figure()
    ax_4 = axes(f4)
    
    cla(ax_4)
    plt.plot(ax_4, pt, height)
    plt.hold(True)
    plt.plot(ax_4, tk, height)
    plt.plot(ax_4, PBLpt, 'g', label = "\theta Method")
    plt.plot(ax_4, PBLri, 'r', label = "Ri Method")
    plt.plot(ax_4, PBLsh, 'b', label = "Specific Hu Method")
    plt.hold(False)
    plt.title('Potential Temperature & Temperature (K)')
    plt.axis(ax_4, [280,320,0,4000])
    plt.show


# In[ ]:




