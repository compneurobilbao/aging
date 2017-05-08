#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:36:19 2017

@author: asier
"""
import h5py
import numpy as np
import os
import glob
from nilearn.plotting import plot_connectome


def calculate_centroid(rois):

    coords = np.loadtxt('/home/asier/git/bha/bha/data/MNI_coords.txt')

    x, y, z = np.zeros(len(rois)), np.zeros(len(rois)), np.zeros(len(rois))

    for i, roi in enumerate(rois):
        x[i], y[i], z[i] = coords[roi].ravel()[0], \
                           coords[roi].ravel()[1], \
                           coords[roi].ravel()[2]

    return np.array([np.mean(x), np.mean(y), np.mean(z)])


def extract_info_from_mat(matfile):

    f = h5py.File(matfile)
    data = np.array(f['netNodeSignSFC']).ravel()

    link_strength = data[np.where(logical_and(data > 0, data != 2))][0]
    sources = np.argwhere(data == 2)
    source_size = len(sources)

    targets = np.argwhere(logical_and(data > 0, data != 2))
    target_size = len(targets)
    
    source_coord = calculate_centroid(sources)
    target_coord = calculate_centroid(targets)
    
    return link_strength, source_coord, source_size, target_coord, target_size
    
    




node_number = len(os.listdir('/home/asier/Desktop/figure6/'))
connectivity_matrix = np.zeros([node_number * 2, node_number * 2])
coords = np.zeros([node_number * 2, 3])
node_size = np.zeros([node_number * 2])

for i, file in enumerate(glob.glob('/home/asier/Desktop/figure6/*')):
    
    link_strength, coords[i*2], node_size[i*2], coords[i*2+1], node_size[i*2+1] = extract_info_from_mat(file)
    connectivity_matrix[i*2][i*2+1] = connectivity_matrix[i*2+1][i*2] =  link_strength
    
    
plot_connectome(adjacency_matrix = connectivity_matrix,
                node_coords = coords,
                node_size = node_size*10,
                node_color= 'k')    
    
    




matfile = '/home/asier/Desktop/figure6/seed_partition_107_module_ 1_target_22_SFC_signLinks.mat'