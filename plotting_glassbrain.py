#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:13:22 2017

@author: asier
"""

from nilearn import image
from nilearn import plotting as plt
import matplotlib.pyplot
import nibabel as nib
import os
import numpy as np

   


max_val = 0
min_val = 0.22
# get maximum value for vmax
for root, dirs, files in os.walk("/home/asier/Desktop/figs"):
	for file in files:
		if file.endswith(".nii"):
			max_i = nib.volumeutils.finite_range(nib.load(os.path.join(root, file)).get_data())[1]
			max_val = max(max_i, max_val)
# get minimum value for vmin
for root, dirs, files in os.walk("."):
	for file in files:
		if file.endswith(".nii"):
			min_i = nib.volumeutils.finite_range(nib.load(os.path.join(root, file)).get_data())[1]
			if min_i: 
				min_val = min(min_i, min_val)


for root, dirs, files in os.walk("/home/asier/Desktop/figs"):
	for file in files:
         if file.endswith(".nii"):
             print(os.path.join(root, file))
             statmap = nib.load(os.path.join(root, file))

             # First plot the map for the PCC: index 4 in the atlas
             plt.plot_glass_brain(statmap, threshold=0,
                                  colorbar=True,
                                  cmap=matplotlib.pyplot.cm.autumn,
                                  display_mode='lyrz', vmax = max_val,
                                  vmin = min_val,
                                  output_file=os.path.join(root,
                                                           file)+
                                                           'glass_white_max.png')
			
 
#statmap = nib.load('/home/asier/Desktop/netMode_SF_intExt_participation_quad.nii')
#plt.plot_glass_brain(statmap, threshold=0,
#                                  colorbar=True,
#                                  cmap=matplotlib.pyplot.cm.autumn,
#                                  display_mode='lyrz', vmax = 3,
#                                  vmin = 1,
#                                  output_file='/home/asier/Desktop/netMode_SF_intExt_participation_quad.png')



ssssssssssssssssssssssssssssssssss









