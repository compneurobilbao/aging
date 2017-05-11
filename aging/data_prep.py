from __future__ import absolute_import, division, print_function
from itertools import product

import os
from os.path import join as opj
import aging as ag
import scipy.io as sio
import numpy as np

data_path = opj(ag.__path__[0], 'data')
aging_data_dir = opj(data_path, 'subjects')
container_data_dir = opj(data_path, 'container_data')  # ID_subj,FCpil
mod_data_dir = opj(data_path, 'mods')


# Create SC and FC matrices
def generate_age_sex_ID():
    age_subj = []
    sex_subj = []
    ID_subj = os.listdir(aging_data_dir)
    ID_subj.sort()

    for idx in ID_subj:
        folder_path = opj(aging_data_dir, idx)
        age = np.loadtxt(opj(folder_path, 'age.txt'))
        sex = np.loadtxt(opj(folder_path, 'gender.txt'), dtype='str')
        age_subj.append(age.tolist())
        sex_subj.append(sex.tolist()[2])

    np.save(opj(container_data_dir, 'age'), age_subj)
    np.save(opj(container_data_dir, 'sex'), sex_subj)
    np.save(opj(container_data_dir, 'ID_subj'), ID_subj)
    return


def generate_FC(ID_subj):

    FC_matrix = np.empty((2514, 2514, len(ID_subj)), dtype='float32')

    for i, idx in enumerate(ID_subj):
        folder_path = opj(aging_data_dir, idx)
        time_series = np.load(opj(folder_path, 'time_series.npy'))
        fc = np.corrcoef(time_series)

        fc = np.nan_to_num(fc)

        FC_matrix[:, :, i] = fc

    return FC_matrix


def generate_SC(ID_subj):

    SC_matrix = np.empty((2514, 2514, len(ID_subj)), dtype='int16')

    for i, idx in enumerate(ID_subj):
        folder_path = opj(aging_data_dir, idx)
        fiber_num = np.load(opj(folder_path, 'fiber_num.npy'))
        # TODO: Try Sparse - not good for concatenation

        SC_matrix[:, :, i] = fiber_num

    return SC_matrix


def generate_data_containers():

    if not os.path.exists(container_data_dir):
        os.makedirs(container_data_dir)

    generate_age_sex_ID()


def remove_corrupted_rois(FC_matrix, SC_matrix):
    ROI_part = np.loadtxt(opj(container_data_dir,
                                       'rois_full_inside_ventricles.txt'),
                          dtype='int')
    ROI_in = np.loadtxt(opj(container_data_dir,
                                     'rois_part_inside_ventricles.txt'),
                        dtype='int')
    ROI_2ex = np.concatenate((ROI_part, ROI_in), axis=0) - 1 # IMPORTANT! Python starts on 0

    FC_matrix[ROI_2ex, :, :] = 0
    FC_matrix[:, ROI_2ex, :] = 0
    SC_matrix[ROI_2ex, :, :] = 0
    SC_matrix[:, ROI_2ex, :] = 0

    return FC_matrix, SC_matrix


def generate_mod(nMod, FC_matrix, SC_matrix, modules_info, ID_subj):
    from numpy import squeeze as sq

    modules_idx = modules_info[nMod-1, :nMod]-1  # -1 due to 0 indexing

    FC_Mod = np.empty((nMod, nMod, len(ID_subj)), dtype='float32')
    SC_Mod = np.empty((nMod, nMod, len(ID_subj)), dtype='float32')

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

    np.savez(opj(mod_data_dir, 'mod_{}'.format(nMod)),
             FC_Mod=FC_Mod, SC_Mod=SC_Mod)


def build_FC_SC_mods():

    if not os.path.exists(mod_data_dir):
        os.makedirs(mod_data_dir)

    ID_subj = np.load(opj(container_data_dir, 'ID_subj.npy'))

    FC_SC_matrix = opj(container_data_dir, 'FC_SC_matrix.npz')
    if not os.path.exists(FC_SC_matrix):
        FC_matrix = generate_FC(ID_subj)
        SC_matrix = generate_SC(ID_subj)
        FC_matrix, SC_matrix = remove_corrupted_rois(FC_matrix, SC_matrix)
        np.savez(FC_SC_matrix, FC_matrix=FC_matrix, SC_matrix=SC_matrix)
    else:
        FC_SC_matrix = np.load(FC_SC_matrix)
        FC_matrix = FC_SC_matrix.f.FC_matrix
        SC_matrix = FC_SC_matrix.f.SC_matrix

    partition_data = sio.loadmat(opj(container_data_dir,
                                              'partition_2_2514.mat'))
    modules_info = partition_data['modules_20_60']

    for nMod in range(2, 1001):
        if not os.path.exists(opj(mod_data_dir,
                                           'mod_{}.npz'.format(nMod))):
            generate_mod(nMod,
                         FC_matrix,
                         SC_matrix,
                         modules_info,
                         ID_subj)


if __name__ == "__main__":
    import sys

    if not os.path.exists(opj(container_data_dir,
                                       'ID_subj.npy')):
        generate_data_containers()
    print('Data containers generated')
    build_FC_SC_mods()
    print('mods generated')
    sys.exit()
