#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:36:19 2017

@author: asier
"""
import os
from os.path import join as opj
import glob
import aging as ag
import scipy
import h5py
import numpy as np

from nilearn.plotting import plot_connectome

DATA_PATH = opj(ag.__path__[0], 'data')
CONTAINER_DATA_PATH = opj(DATA_PATH, 'container_data')


def calculate_centroid(rois):

    mni_coords = np.loadtxt(opj(CONTAINER_DATA_PATH,
                                'MNI_coords.txt'))

    x, y, z = np.zeros(len(rois)), np.zeros(len(rois)), np.zeros(len(rois))

    for i, roi in enumerate(rois):
        x[i], y[i], z[i] = mni_coords[roi].ravel()[0], \
                           mni_coords[roi].ravel()[1], \
                           mni_coords[roi].ravel()[2]
    # corrected MNI in x axis
    return np.array([np.mean(-x), np.mean(y), np.mean(z)])


def calculate_size(roi_list):

    file = scipy.io.loadmat(opj(CONTAINER_DATA_PATH, 'rois_size.mat'))
    rois_to_aal = file['rois_size'].ravel()

    return sum([rois_to_aal[roi][0] for roi in roi_list])


def extract_info_from_mat(matfile):

    file = h5py.File(matfile)
    data = np.array(file['netNodeSignSFC']).ravel()

    link_strength = data[np.where(np.logical_and(data > 0, data != 2))][0]
    sources = np.argwhere(data == 2)
    source_size = calculate_size(sources)

    targets = np.argwhere(np.logical_and(data > 0, data != 2))
    target_size = calculate_size(targets)

    source_coord = calculate_centroid(sources)
    target_coord = calculate_centroid(targets)

    return link_strength, source_coord, source_size, target_coord, target_size


def coord_to_aal(coord):

    # AALLabelID116 and modules_aal
    file = scipy.io.loadmat(opj(CONTAINER_DATA_PATH,
                                'modules_aal_labels.mat'))
    rois_to_aal = file['modules_aal'].ravel()
    aal_names = ['not recognized'] + \
                [label[1][0] for label in file['AALLabelID116']]
    mni_coords = np.loadtxt(opj(CONTAINER_DATA_PATH,
                                'MNI_coords.txt'))

    coord = np.round(coord)
    dist = [np.linalg.norm(mni_coord-coord) for mni_coord in mni_coords]

    return aal_names[rois_to_aal[np.argmin(dist)]]


if __name__ == "__main__":
    node_number = len(os.listdir('/home/asier/Desktop/figure6/'))
    connectivity_matrix = np.zeros([node_number * 2, node_number * 2])
    coords = np.zeros([node_number * 2, 3])
    node_size = np.zeros([node_number * 2])

    for i, file in enumerate(glob.glob('/home/asier/Desktop/figure6/*')):

        link_strength, coords[i*2], node_size[i*2], coords[i*2+1], node_size[i*2+1] = extract_info_from_mat(file)
        connectivity_matrix[i*2][i*2+1] = connectivity_matrix[i*2+1][i*2] = link_strength

    plot_connectome(adjacency_matrix=connectivity_matrix,
                    node_coords=coords,
                    node_size=node_size,
                    node_color='auto')
    # AAL strutctures information
    for coord in coords:
        print(coord_to_aal(coord))
    # Structures size information
    for size in node_size:
        print(size*27)     # 3mm voxel. 3^3 mm^3
