# -*- coding: utf-8 -*-
"""
Project: Particle Image Velocimetry (PIV) code!
         Sample code to call the function!
@author: A. F. Forughi (Aug. 2020, Last update: Jun. 2021)
"""

# %% Libraries:
import numpy as np
import cv2
import matplotlib.pyplot as plt
from piv_lib import piv # Importing the PIV function


# %% Loading the images:
img_1 = (np.flip(cv2.imread('image_01.tif', 0),0)).astype('float32') # Read Grayscale
img_2 = (np.flip(cv2.imread('image_02.tif', 0),0)).astype('float32')


# %% Setting the PIV parameters:
iw=21 # Interrodation Windows Sizes (pixel)
sw=81 # Search Windows Sizes (sw > iw) (pixel)  

r_limit=0.5   # minimum acceptable correlation coefficient. If you're not sure start with 0.6

i_fix=500     # Number of maximum correction cycles ; 0 means no correction

l_scale=1.0   # spatial scale [m/pixel] ; 1 means no size scaling
t_scale=1.0   # time step = 1/frame_rate [s/frame] ; 1 means no time scaling


# %% Runing PIV function:
""" *** Here are the function's arguments and returned values ***
    
Arguments: 
    first image as a Numpy matrix (img_1)
    first image as a Numpy matrix (img_2)
    size of the introgation window (IW)
    Size of the search window (SW)
    Minimum acceptable correlation coefficient (r_limit)
    Number of maximum correction cycles (i_fix)
    Spatial scale [m/pixel]
    Time step = 1/frame_rate [s/frame]
    
Returned values:
    X Position of the vectors (X)
    Y Position of the vectors (Y)
    X-velocity components (vecx)
    Y-velocity components (vecy)
    Velocity vector size (vec)
    Correlation coefficient of each intoregation window (rij)
"""

X, Y, vecx, vecy, vec, rij = piv(img_1,img_2,iw,sw,r_limit,i_fix,l_scale,t_scale)


# %% Exporting Data in as a Numpy file:
np.savez('results.npz', X=X, Y=Y, vecx=vecx, vecy=vecy, vec=vec, rij=rij)

# res=np.load('results.npz'); X=res['X']; Y=res['Y']; vecx=res['vecx']; vecy=res['vecy']; vec=res['vec']; rij=res['rij']; # Load saved data


# %% Generating graphs:
    
ia,ja = img_1.shape
fig, ax = plt.subplots(figsize=(8,8*ia/ja), dpi=300)
q = ax.quiver(X, Y, vecx, vecy,units='width')
plt.show()


fig, ax = plt.subplots(figsize=(8,8*ia/ja), dpi=300)
plt.contourf(X[0],np.transpose(Y)[0],rij,cmap='jet',levels=np.arange(rij.min(),min(rij.max()+0.1,1.0),0.01))
plt.colorbar(label='R')
plt.show()

fig, ax = plt.subplots(figsize=(8,8*ia/ja), dpi=300)
plt.streamplot(X, Y, vecx, vecy,density=3,linewidth=0.8,color=vec)
plt.colorbar(label='Velocity')
plt.show()
