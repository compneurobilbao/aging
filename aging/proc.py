from __future__ import absolute_import, division, print_function

import os
from os.path import join as opj
import aging as ag
import numpy as np
from scipy import stats
import pickle

DATA_PATH = opj(ag.__path__[0], 'data')
aging_data_dir = opj(DATA_PATH, 'subjects')
CONTAINER_DATA_DIR = opj(DATA_PATH, 'container_data')  # ID_subj,FCpil
MOD_DATA_DIR = opj(DATA_PATH, 'mods')


MAX_PART = 1000


def p_corr(x, y, z):
    # PARTCORRCOEF calculates the partial correlation between X and Y
    # after removing the influence of Z.

    #   ADAPTED FROM:
    #    $Id: partcorrcoef.m 8351 2011-06-24 17:35:07Z carandraug $
    #    Copyright(C)2000-2002,2009 by Alois Schloegl alois.schloegl@gmail.com
    #    This function is part of the NaN-toolbox
    #    http://pub.ist.ac.at/~schloegl/matlab/NaN/

    #    This program is free software; you can redistribute it and/or modify
    #    it under the terms of the GNU General Public License as published by
    #    the Free Software Foundation; either version 3 of the License, or
    #    (at your option) any later version.
    #
    #    This program is distributed in the hope that it will be useful,
    #    but WITHOUT ANY WARRANTY; without even the implied warranty of
    #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #    GNU General Public License for more details.
    #
    #    You should have received a copy of the GNU General Public License
    #    along with this program; if not, write to the FSF
    #    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA

    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]

    c = (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))

    # SIGNIFICANCE TEST
    NN = x.shape[0] - 1
    tmp = 1 - c * c

    if tmp < 0:
        tmp = 0  # prevent tmp<0 i.e. imag(t)~=0

    t = c * np.sqrt(np.max((NN-2, 0)) / tmp)
    v = stats.t.cdf(t, NN-2)
    v = 2 * np.min((v, 1 - v))

    return c, v


def partial(Mod_data, age, motion):
    value, sig = p_corr(Mod_data, age, motion)
    return np.nan_to_num(value), np.nan_to_num(sig)


def generate_dti_fmri_motion():
    dti_motion = []
    fmri_motion = []
    ID_subj = os.listdir(aging_data_dir)
    ID_subj.sort()

    for idx in ID_subj:
        folder_path = opj(aging_data_dir, idx)
        dti_motion_subject = np.load(opj(folder_path,
                                         'dti_motion.npy')).astype(np.float)
        fmri_motion_subject = np.load(opj(folder_path,
                                          'fmri_motion.npy')).astype(np.float)
        dti_motion.append(dti_motion_subject)
        fmri_motion.append(fmri_motion_subject)

    np.save(opj(CONTAINER_DATA_DIR, 'dti_motion'), dti_motion)
    np.save(opj(CONTAINER_DATA_DIR, 'fmri_motion'), fmri_motion)
    return


def init_variables():

    if not os.path.exists(opj(CONTAINER_DATA_DIR, 'fmri_motion.npy')):
        generate_dti_fmri_motion()

    age = np.load(opj(CONTAINER_DATA_DIR, 'age.npy'))
    dti_motion = np.load(opj(CONTAINER_DATA_DIR, 'dti_motion.npy'))
    fmri_motion = np.load(opj(CONTAINER_DATA_DIR, 'fmri_motion.npy'))

    return age, dti_motion, fmri_motion


# better to compute then together: loading the mods takes most of the time
def compute_connectivity(internal=False, external=False):
    if not internal and not external:
        print('Not computing anything')
        return

    age, dti_motion, fmri_motion = init_variables()

    if internal:
        int_fc_cn = np.array([np.zeros(j) for j in range(MAX_PART)])
        int_sc_cn = np.array([np.zeros(j) for j in range(MAX_PART)])
        int_fc_pn = np.array([np.zeros(j) for j in range(MAX_PART)])
        int_sc_pn = np.array([np.zeros(j) for j in range(MAX_PART)])
    if external:
        ext_fc_cn = np.array([np.zeros(j) for j in range(MAX_PART)])
        ext_sc_cn = np.array([np.zeros(j) for j in range(MAX_PART)])
        ext_fc_pn = np.array([np.zeros(j) for j in range(MAX_PART)])
        ext_sc_pn = np.array([np.zeros(j) for j in range(MAX_PART)])

    for nMod in range(2, 999):
        print(nMod)
        data = np.load(opj(MOD_DATA_DIR, 'mod_{}.npz'.format(nMod)))
        FC_Mod = data.f.FC_Mod
        SC_Mod = data.f.SC_Mod

        if internal:
            int_fc_cn[nMod] = np.zeros(nMod)
            int_fc_pn[nMod] = np.zeros(nMod)
            int_sc_cn[nMod] = np.zeros(nMod)
            int_sc_pn[nMod] = np.zeros(nMod)
        if external:
            ext_fc_cn[nMod] = np.zeros(nMod)
            ext_fc_pn[nMod] = np.zeros(nMod)
            ext_sc_cn[nMod] = np.zeros(nMod)
            ext_sc_pn[nMod] = np.zeros(nMod)
            SC_Mod_total_degree = np.sum(SC_Mod, 1)
            FC_Mod_total_degree = np.sum(FC_Mod, 1)

        for i in range(nMod):
            if internal:
                int_fc_cn[nMod][i], int_fc_pn[nMod][i] = \
                    partial(FC_Mod[i, i, :], age, fmri_motion)
                int_sc_cn[nMod][i], int_sc_pn[nMod][i] = \
                    partial(SC_Mod[i, i, :], age, dti_motion)
            if external:
                FC_data = FC_Mod_total_degree[i, :] - FC_Mod[i, i, :]
                SC_data = SC_Mod_total_degree[i, :] - SC_Mod[i, i, :]
                ext_fc_cn[nMod][i], ext_fc_pn[nMod][i] = \
                    partial(FC_data, age, fmri_motion)
                ext_sc_cn[nMod][i], ext_sc_pn[nMod][i] = \
                    partial(SC_data, age, dti_motion)

    if internal:
        with open(opj(CONTAINER_DATA_DIR, 'internal_fc_cn_pn'), 'wb') as f:
            pickle.dump([int_fc_cn, int_fc_pn], f)
        with open(opj(CONTAINER_DATA_DIR, 'internal_sc_cn_pn'), 'wb') as f:
            pickle.dump([int_sc_cn, int_sc_pn], f)
    if external:
        with open(opj(CONTAINER_DATA_DIR, 'external_fc_cn_pn'), 'wb') as f:
            pickle.dump([ext_fc_cn, ext_fc_pn], f)
        with open(opj(CONTAINER_DATA_DIR, 'external_sc_cn_pn'), 'wb') as f:
            pickle.dump([ext_sc_cn, ext_sc_pn], f)


def get_descriptors():

    int_fc_cn, int_fc_pn = pickle.load(open(opj(CONTAINER_DATA_DIR,
                                                'internal_fc_cn_pn'), 'rb'))
    ext_fc_cn, ext_fc_pn = pickle.load(open(opj(CONTAINER_DATA_DIR,
                                                'external_fc_cn_pn'), 'rb'))

    int_sc_cn, int_sc_pn = pickle.load(open(opj(CONTAINER_DATA_DIR,
                                                'internal_sc_cn_pn'), 'rb'))
    ext_sc_cn, ext_sc_pn = pickle.load(open(opj(CONTAINER_DATA_DIR,
                                                'external_sc_cn_pn'), 'rb'))

    # check pvalues and get descriptors from nMod npz
    for i, values_list in enumerate(ext_fc_cn[2:]):
        if max(values_list) > 0.34:
            print(max(values_list),
                  ext_fc_pn[i+2][np.argmax(np.array(values_list))])
