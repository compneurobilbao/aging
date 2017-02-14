from __future__ import absolute_import, division, print_function
from itertools import product

import os
import aging as ag
import numpy as np
from scipy import stats, linalg
import pickle

data_path = os.path.join(ag.__path__[0], 'data')
aging_data_dir = os.path.join(data_path, 'subjects')
container_data_dir = os.path.join(data_path, 'container_data')  # ID_subj,FCpil
mod_data_dir = os.path.join(data_path, 'mods')

from .due import due, Doi

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi(""),
         description="",
         tags=[""],
         path='aging')

MAX_PART = 2514




def p_corr(C):
    """
    Partial Correlation in Python (clone of Matlab's partialcorr)
    This uses the linear regression approach to compute the partial 
    correlation (might be slow for a huge number of variables). The 
    algorithm is detailed here:
        http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
    the algorithm can be summarized as
        1) perform a normal linear least-squares regression with X as the target and Z as the predictor
        2) calculate the residuals in Step #1
        3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
        4) calculate the residuals in Step #3
        5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 
        The result is the partial correlation between X and Y while controlling for the effect of Z
    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    """
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr


def partial(Mod_data, age, motion):
    value = p_corr(np.column_stack((Mod_data, age, motion)))[0, 1]
    return np.nan_to_num(value)
                             

def internal():
    # TODO: Significant pvlues for partial_corr
    internal_fc_cn = ['' for j in range(MAX_PART)]
    internal_sc_cn = ['' for j in range(MAX_PART)]

    age = np.load(os.path.join(container_data_dir, 'age.npy'))
    dti_motion = np.load(os.path.join(container_data_dir, 'dti_motion.npy'))
    fmri_motion = np.load(os.path.join(container_data_dir, 'fmri_motion.npy'))

    for nMod in range(2, 999):
        print(nMod)
        data = np.load(os.path.join(mod_data_dir, 'mod_{}.npz'.format(nMod)))
        FC_Mod = data.f.FC_Mod
        SC_Mod = data.f.SC_Mod

        internal_fc_cn[nMod] = np.zeros(nMod)
        internal_sc_cn[nMod] = np.zeros(nMod)

        for i in range(nMod):
            # (out-degree)
            internal_fc_cn[nMod][i] = partial(FC_Mod[i, i, :],
                                              age,
                                              fmri_motion)

            internal_sc_cn[nMod][i] = partial(SC_Mod[i, i, :],
                                              age,
                                              dti_motion)

    with open(os.path.join(container_data_dir, 'internal_fc_cn'), "wb") as f:
        pickle.dump(internal_fc_cn, f)
    with open(os.path.join(container_data_dir, 'internal_sc_cn'), "wb") as f:
        pickle.dump(internal_sc_cn, f)


def external():
    # TODO: Significant pvlues for partial_corr
    external_fc_cn = ['' for j in range(MAX_PART)]
    external_sc_cn = ['' for j in range(MAX_PART)]

    age = np.load(os.path.join(container_data_dir, 'age.npy'))
    dti_motion = np.load(os.path.join(container_data_dir, 'dti_motion.npy'))
    fmri_motion = np.load(os.path.join(container_data_dir, 'fmri_motion.npy'))

    for nMod in range(2, 999):
        print(nMod)
        data = np.load(os.path.join(mod_data_dir, 'mod_{}.npz'.format(nMod)))
        FC_Mod = data.f.FC_Mod
        SC_Mod = data.f.SC_Mod

        external_fc_cn[nMod] = np.zeros(nMod)
        external_sc_cn[nMod] = np.zeros(nMod)

        SC_Mod_total_degree = np.sum(SC_Mod, 1)
        FC_Mod_total_degree = np.sum(SC_Mod, 1)

        for i in range(nMod):
            # (out-degree)
            FC_data = FC_Mod_total_degree[i, :] - FC_Mod[i, i, :]
            external_fc_cn[nMod][i] = partial(FC_data,
                                              age,
                                              fmri_motion)

            SC_data = SC_Mod_total_degree[i, :] - SC_Mod[i, i, :]
            external_sc_cn[nMod][i] = partial(SC_data,
                                              age,
                                              dti_motion)

    with open(os.path.join(container_data_dir, 'internal_fc_cn'), "wb") as f:
        pickle.dump(external_fc_cn, f)
    with open(os.path.join(container_data_dir, 'internal_sc_cn'), "wb") as f:
        pickle.dump(external_sc_cn, f)
