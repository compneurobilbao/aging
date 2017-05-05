#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:50:26 2017

@author: asier
"""

from numpy import squeeze as sq
from scipy.io import loadmat
partition_data = sio.loadmat(os.path.join(container_data_dir,
                                              'partition_2_2514.mat'))
modules_info = partition_data['modules_20_60']
nMod = 20

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
    FC_Mod[i, j, :] = _A
    FC_Mod[j, i, :] = FC_Mod[i, j, :]

FC_Mod[0,0,0]


for nMod in range(2, 1001):
    print(nMod)
    if not os.path.exists(os.path.join(mod_data_dir,
                                       'mod_{}.npz'.format(nMod))):
        generate_mod(nMod,
                     FC_matrix,
                     SC_matrix,
                     modules_info,
                     ID_subj)

compute_connectivity(internal=True, external=True)