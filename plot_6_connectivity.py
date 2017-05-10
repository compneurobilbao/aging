#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:36:19 2017

@author: asier
"""
import aging as ag
import scipy
import h5py
import numpy as np
import os
import glob
from nilearn.plotting import plot_connectome

data_path = os.path.join(ag.__path__[0], 'data')
container_data_dir = os.path.join(data_path, 'container_data')


def calculate_centroid(rois):

    mni_coords = np.loadtxt(os.path.join(container_data_dir, 'MNI_coords.txt'))

    x, y, z = np.zeros(len(rois)), np.zeros(len(rois)), np.zeros(len(rois))

    for i, roi in enumerate(rois):
        x[i], y[i], z[i] = mni_coords[roi].ravel()[0], \
                           mni_coords[roi].ravel()[1], \
                           mni_coords[roi].ravel()[2]

    return np.array([np.mean(-x), np.mean(y), np.mean(z)]) # corrected MNI in x axis


def calculate_size(roi_list):
    
    f = scipy.io.loadmat(os.path.join(container_data_dir, 'rois_size.mat'))
    rois_to_aal = f['rois_size'].ravel()

    return sum([rois_to_aal[roi][0] for roi in roi_list])

def extract_info_from_mat(matfile):

    f = h5py.File(matfile)
    data = np.array(f['netNodeSignSFC']).ravel()

    link_strength = data[np.where(logical_and(data > 0, data != 2))][0]
    sources = np.argwhere(data == 2)
    source_size = calculate_size(sources)

    targets = np.argwhere(logical_and(data > 0, data != 2))
    target_size = calculate_size(targets)
    
    source_coord = calculate_centroid(sources)
    target_coord = calculate_centroid(targets)
    
    return link_strength, source_coord, source_size, target_coord, target_size
    
    
#def coords_to_AAL(coords):
#    
#    from nilearn.datasets import fetch_atlas_aal
#    import nibabel as nb
#
#    np.set_printoptions(precision=3, suppress=True)
#    
#    atlas = fetch_atlas_aal()
#    atlas_filename = atlas.maps
#    img = nb.load(atlas_filename)
#    
#    voxel_coords = np.round(f(img, coords))
#    
#    image_data = img.get_data()
#
#def f(img, coords):
#    """ Return X, Y, Z coordinates for i, j, k """
#
#    M = img.affine[:3, :3]
#    abc = img.affine[:3, 3]
#
#    return M.dot(coords) + abc


def coord_to_AAL(coord):
    
    # AALLabelID116 and modules_aal
    f = scipy.io.loadmat(os.path.join(container_data_dir, 'modules_aal_labels.mat'))
    rois_to_aal = f['modules_aal'].ravel()
    aal_names = ['not recognized'] + [label[1][0] for label in f['AALLabelID116']]
    mni_coords = np.loadtxt(os.path.join(container_data_dir, 'MNI_coords.txt'))
    
    dist = np.zeros(len(mni_coords))
    coord = np.round(coord)

    for i, mni_coord in enumerate(mni_coords):
        dist[i] = numpy.linalg.norm(mni_coord-coord)

    return aal_names[rois_to_aal[np.argmin(dist)]]


if __name__ == "__main__":
    
    node_number = len(os.listdir('/home/asier/Desktop/figure6/'))
    connectivity_matrix = np.zeros([node_number * 2, node_number * 2])
    coords = np.zeros([node_number * 2, 3])
    node_size = np.zeros([node_number * 2])
    
    for i, file in enumerate(glob.glob('/home/asier/Desktop/figure6/*')):
        
        link_strength, coords[i*2], node_size[i*2], coords[i*2+1], node_size[i*2+1] = extract_info_from_mat(file)
        connectivity_matrix[i*2][i*2+1] = connectivity_matrix[i*2+1][i*2] =  link_strength
        
        
    plot_connectome(adjacency_matrix = connectivity_matrix,
                    node_coords = coords,
                    node_size = node_size,
                    node_color= 'auto')    
    
    for coord in coords:
        print(coord_to_AAL(coord))
        
    for size in node_size:
        print(size*27)     # 3mm voxel. 3^3 mm^3

        