#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:36:19 2017

@author: asier
"""
from __future__ import absolute_import, division, print_function
from itertools import product

import os
from os.path import join as opj
import aging as ag
import scipy.io as sio
import numpy as np

DATA_PATH = opj(ag.__path__[0], 'data')
AGING_DATA_DIR = opj(DATA_PATH, 'subjects')
CONTAINER_DATA_DIR = opj(DATA_PATH, 'container_data')  # id_subj,FCpil
MOD_DATA_DIR = opj(DATA_PATH, 'mods')


# Create SC and FC matrices
def generate_age_sex_id():
    age_subj = []
    sex_subj = []
    id_subj = os.listdir(AGING_DATA_DIR)
    id_subj.sort()

    for idx in id_subj:
        folder_path = opj(AGING_DATA_DIR, idx)
        age = np.loadtxt(opj(folder_path, 'age.txt'))
        sex = np.loadtxt(opj(folder_path, 'gender.txt'), dtype='str')
        age_subj.append(age.tolist())
        sex_subj.append(sex.tolist()[2])

    np.save(opj(CONTAINER_DATA_DIR, 'age'), age_subj)
    np.save(opj(CONTAINER_DATA_DIR, 'sex'), sex_subj)
    np.save(opj(CONTAINER_DATA_DIR, 'id_subj'), id_subj)
    return


def generate_FC(id_subj):

    FC_matrix = np.empty((2514, 2514, len(id_subj)), dtype='float32')

    for i, idx in enumerate(id_subj):
        folder_path = opj(AGING_DATA_DIR, idx)
        time_series = np.load(opj(folder_path, 'time_series.npy'))
        fc = np.corrcoef(time_series)

        fc = np.nan_to_num(fc)

        FC_matrix[:, :, i] = fc

    return FC_matrix


def generate_SC(id_subj):

    SC_matrix = np.empty((2514, 2514, len(id_subj)), dtype='int16')

    for i, idx in enumerate(id_subj):
        folder_path = opj(AGING_DATA_DIR, idx)
        fiber_num = np.load(opj(folder_path, 'fiber_num.npy'))
        # TODO: Try Sparse - not good for concatenation

        SC_matrix[:, :, i] = fiber_num

    return SC_matrix


def generate_data_containers():

    if not os.path.exists(CONTAINER_DATA_DIR):
        os.makedirs(CONTAINER_DATA_DIR)

    generate_age_sex_id()


def remove_corrupted_rois(FC_matrix, SC_matrix):
    roi_part = np.loadtxt(opj(CONTAINER_DATA_DIR,
                              'rois_full_inside_ventricles.txt'),
                          dtype='int')
    roi_in = np.loadtxt(opj(CONTAINER_DATA_DIR,
                            'rois_part_inside_ventricles.txt'),
                        dtype='int')

    # -1 due to python indexing and roi = 0 not existing
    roi_2ex = np.concatenate((roi_part, roi_in), axis=0) - 1

    FC_matrix[roi_2ex, :, :] = 0
    FC_matrix[:, roi_2ex, :] = 0
    SC_matrix[roi_2ex, :, :] = 0
    SC_matrix[:, roi_2ex, :] = 0

    return FC_matrix, SC_matrix


def generate_mod(nMod, FC_matrix, SC_matrix, modules_info, id_subj):
    from numpy import squeeze as sq

    modules_idx = modules_info[nMod-1, :nMod]-1  # -1 due to 0 indexing

    FC_Mod = np.empty((nMod, nMod, len(id_subj)), dtype='float32')
    SC_Mod = np.empty((nMod, nMod, len(id_subj)), dtype='float32')

    for i, j in product(range(nMod), range(nMod)):
        if modules_idx[i].shape[0] == 1 or modules_idx[j].shape[0] == 1:
            idx_i, idx_j = sq(modules_idx[i]), sq(modules_idx[j])
        else:
            idx_i, idx_j = np.ix_(sq(modules_idx[i]), sq(modules_idx[j]))

        _A = FC_matrix[idx_i, idx_j, :]
        while _A.ndim != 1:
            _A = np.sum(_A, 0)
        FC_Mod[i, j, :] = _A  # / (len(modules_idx[i]) * len(modules_idx[j]))
        FC_Mod[j, i, :] = FC_Mod[i, j, :]

        _B = SC_matrix[idx_i, idx_j, :]
        while _B.ndim != 1:
            _B = np.sum(_B, 0)
        SC_Mod[i, j, :] = _B  # / (len(modules_idx[i]) * len(modules_idx[j]))
        SC_Mod[j, i, :] = SC_Mod[i, j, :]

    np.savez(opj(MOD_DATA_DIR, 'mod_{}'.format(nMod)),
             FC_Mod=FC_Mod, SC_Mod=SC_Mod)


def build_FC_SC_mods():

    if not os.path.exists(MOD_DATA_DIR):
        os.makedirs(MOD_DATA_DIR)

    id_subj = np.load(opj(CONTAINER_DATA_DIR, 'id_subj.npy'))

    FC_SC_matrix_path = opj(CONTAINER_DATA_DIR, 'FC_SC_matrix.npz')
    if not os.path.exists(FC_SC_matrix_path):
        FC_matrix = generate_FC(id_subj)
        SC_matrix = generate_SC(id_subj)
        FC_matrix, SC_matrix = remove_corrupted_rois(FC_matrix, SC_matrix)
        np.savez(FC_SC_matrix_path, FC_matrix=FC_matrix, SC_matrix=SC_matrix)
    else:
        FC_SC_matrix = np.load(FC_SC_matrix_path)
        FC_matrix = FC_SC_matrix.f.FC_matrix
        SC_matrix = FC_SC_matrix.f.SC_matrix

    partition_data = sio.loadmat(opj(CONTAINER_DATA_DIR,
                                     'partition_2_2514.mat'))
    modules_info = partition_data['modules_20_60']

    for nMod in range(2, 1001):
        if not os.path.exists(opj(MOD_DATA_DIR,
                                  'mod_{}.npz'.format(nMod))):
            generate_mod(nMod,
                         FC_matrix,
                         SC_matrix,
                         modules_info,
                         id_subj)


if __name__ == "__main__":
    import sys

    if not os.path.exists(opj(CONTAINER_DATA_DIR,
                              'id_subj.npy')):
        generate_data_containers()
    print('Data containers generated')
    build_FC_SC_mods()
    print('mods generated')
    sys.exit()
