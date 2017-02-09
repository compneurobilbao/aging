from __future__ import absolute_import, division, print_function
# Ugly hack to allow absolute import from the root folder
# whatever its name is. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))

from distutils.dir_util import copy_tree
from itertools import product

import aging as ag
import scipy.io as sio
import numpy as np
import multiprocessing
import ctypes


data_path = os.path.join(ag.__path__[0], 'data')
aging_data_dir = os.path.join(data_path, 'subjects')
container_data_dir = os.path.join(data_path, 'container_data')  # ID_subj,FCpil
mod_data_dir = os.path.join(data_path, 'mods')

life_span_path = "/home/asier/Desktop/AGING/life_span_paolo"
life_span2_path = "/home/asier/Desktop/AGING/life_span2"


def add_info_from_file(info_file, old_or_young='old'):

    for i in range(len(info_file[old_or_young+'Info'])):
        dr = os.path.join(aging_data_dir,
                          info_file[old_or_young+'Info'][i][0][0])
        try:
            np.savetxt(os.path.join(dr, 'age.txt'),
                       info_file[old_or_young+'Age'][i][0][0],
                       fmt='%s')
            np.savetxt(os.path.join(dr, 'gender.txt'),
                       info_file[old_or_young+'Sex'][i][0],
                       fmt='%s')
        except:
            print('Failed {}'.format(info_file[old_or_young+'Info'][i][0][0]))


def copy_full_tree(src, dst):
    for root, dirs, files in os.walk(src):
        for dr in dirs:
            os.makedirs(os.path.join(dst, dr))
    copy_tree(src, dst)


# Module for data preparation
def order_data():
    copy_full_tree(life_span2_path, aging_data_dir)

    info = sio.loadmat(os.path.join(aging_data_dir, 'partecipantsInfo_v2.mat'))

    add_info_from_file(info, 'old')
    add_info_from_file(info, 'young')

    for file in os.listdir(aging_data_dir):
        if file.endswith(".mat") or file.endswith(".xls"):
            os.remove(os.path.join(aging_data_dir, file))

    for root, dirs, files in os.walk(aging_data_dir):
        for file in files:
            if file.endswith('networks.mat'):
                os.rename(os.path.join(root, file),
                          os.path.join(root, 'time_series.mat'))

    copy_full_tree(life_span_path, aging_data_dir)


# Create SC and FC matrices
def generate_age_sex_ID():
    age_subj = []
    sex_subj = []
    ID_subj = os.listdir(aging_data_dir)
    ID_subj.sort()

    for idx in ID_subj:
        folder_path = os.path.join(aging_data_dir, idx)
        age = np.loadtxt(os.path.join(folder_path, 'age.txt'))
        sex = np.loadtxt(os.path.join(folder_path, 'gender.txt'), dtype='str')
        age_subj.append(age.tolist())
        sex_subj.append(sex.tolist()[2])

    np.save(os.path.join(container_data_dir, 'age'), age_subj)
    np.save(os.path.join(container_data_dir, 'sex'), sex_subj)
    np.save(os.path.join(container_data_dir, 'ID_subj'), ID_subj)
    return


def generate_FC():
    ID_subj = np.load(os.path.join(container_data_dir, 'ID_subj.npy'))

    FC_matrix = np.empty((2514, 2514, len(ID_subj)), dtype='float32')

    for i, idx in enumerate(ID_subj):
        folder_path = os.path.join(aging_data_dir, idx)
        time_series = sio.loadmat(os.path.join(folder_path, 'time_series.mat'))
        fc = np.corrcoef(time_series['time_series'].T)

        fc = np.nan_to_num(fc)

        FC_matrix[:, :, i] = fc

    return FC_matrix


def generate_SC():
    ID_subj = np.load(os.path.join(container_data_dir, 'ID_subj.npy'))

    SC_matrix = np.empty((2514, 2514, len(ID_subj)), dtype='float32')

    for i, idx in enumerate(ID_subj):
        folder_path = os.path.join(aging_data_dir, idx)
        fiber_num = sio.loadmat(os.path.join(folder_path, 'fiber_num.mat'))
        # TODO: Try Sparse - not good for concatenation

        SC_matrix[:, :, i] = fiber_num['fiber_num']

    return SC_matrix


def generate_data_containers():

    if not os.path.exists(container_data_dir):
        os.makedirs(container_data_dir)

    generate_age_sex_ID()


def generate_mod(nMod, FC_matrix, SC_matrix, modules_ordered, ID_subj):
    from numpy import squeeze as sq

    modules_idx = modules_ordered[nMod-1, :nMod]-1  # -1 due to 0 indexing
    'FC & SC descriptors calculations'
    FC_Mod = np.empty((nMod, nMod, len(ID_subj)), dtype='float32')
    SC_Mod = np.empty((nMod, nMod, len(ID_subj)), dtype='float32')

    for i, j in product(range(nMod), range(nMod)):
        idx_i, idx_j = np.ix_(sq(modules_idx[i]), sq(modules_idx[j]))

        _A = FC_matrix[idx_i, idx_j, :]
        FC_Mod[i, j, :] = np.sum(_A, (0, 1)) / (len(modules_idx[i]) * len(modules_idx[j]))
        FC_Mod[j, i, :] = FC_Mod[i, j, :]

        _B = SC_matrix[idx_i, idx_j, :]
        SC_Mod[i, j, :] = np.sum(_B, (0, 1)) / (len(modules_idx[i]) * len(modules_idx[j]))
        SC_Mod[j, i, :] = SC_Mod[i, j, :]

    np.savez(os.path.join(mod_data_dir, 'mod_{}'.format(nMod)),
             'FC_Mod', 'SC_Mod')


def build_FC_SC_mods():

    if not os.path.exists(mod_data_dir):
        os.makedirs(mod_data_dir)

    sh_ID_subj = multiprocessing.Array(ctypes.c_double, (164))
    sh_ID_subj = np.load(os.path.join(container_data_dir, 'ID_subj.npy'))

    sh_FC_matrix = multiprocessing.Array(ctypes.c_double,
                                         (2514, 2514, len(sh_ID_subj)))
    sh_SC_matrix = multiprocessing.Array(ctypes.c_double,
                                         (2514, 2514, len(sh_ID_subj)))
    sh_FC_matrix = generate_FC()
    sh_SC_matrix = generate_SC()

    sh_modules_ordered = multiprocessing.Array(ctypes.c_double,
                                               (2514, 2514))
    partition_data = sio.loadmat(os.path.join(container_data_dir,
                                              'partition_ordered.mat'))
    sh_modules_ordered = partition_data['modules_ordered']

    jobs = []
    for i in range(1, 1001):
        p = multiprocessing.Process(target=generate_mod,
                                    args=(i,
                                          sh_FC_matrix,
                                          sh_SC_matrix,
                                          sh_modules_ordered,
                                          sh_ID_subj))
        jobs.append(p)
        p.start()

if __name__ == "__main__":
    import sys

    generate_data_containers()
    print('Data containers generated')
    build_FC_SC_mods()
    print('mods generated')
    sys.exit()
