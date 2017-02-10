from __future__ import absolute_import, division, print_function
# Ugly hack to allow absolute import from the root folder
# whatever its name is. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    import sys, os
    os.chdir(os.path.realpath(__file__))
    sys.path.insert(0, os.path.abspath('..'))

from distutils.dir_util import copy_tree
from itertools import product

import aging as ag
import scipy.io as sio
import numpy as np


data_path = os.path.join(ag.__path__[0], 'data')
aging_data_dir = os.path.join(data_path, 'subjects')
container_data_dir = os.path.join(data_path, 'container_data')  # ID_subj,FCpil
mod_data_dir = os.path.join(data_path, 'mods')


life_span_path = "/home/asier/Desktop/AGING/life_span_paolo"
life_span2_path = "/home/asier/Desktop/AGING/life_span2"


def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

    
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


# Module for data preparation (OLD)
def order_data_old():
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


# Module for data preparation (NEW) - Modify the OLD one
def order_data_new():

    gsr_sc = '/home/asier/Desktop/aging_final_data/sc_networks'
    gsr_fc = '/home/asier/Desktop/aging_final_data/ts_gsr'

    dst_dir = os.listdir(aging_data_dir)
    dst_dir.sort()

    src_dir_fc = os.listdir(gsr_fc)
    src_dir_fc.sort()

    # src_dir_sc = os.listdir(gsr_sc)
    # src_dir_sc.sort()
    # src_dir_sc == src_dir_fc == True

    for i, file in enumerate(src_dir_fc):
        print(file)
        dst_path = os.path.join(aging_data_dir, dst_dir[i], 'time_series.npy')
        if not os.path.exists(dst_path):
            fc_matrix = np.loadtxt(os.path.join(gsr_fc, file), dtype='float32')
            np.save(os.path.join(aging_data_dir,
                                 dst_dir[i],
                                 'time_series.npy'), fc_matrix)
            silent_remove(os.path.join(aging_data_dir,
                                       dst_dir[i],
                                       'time_series.mat'))

        dst_path = os.path.join(aging_data_dir, dst_dir[i], 'fiber_num.npy')
        if not os.path.exists(dst_path):
            sc_matrix = np.loadtxt(os.path.join(gsr_sc, file), dtype='int16')
            np.save(os.path.join(aging_data_dir,
                                 dst_dir[i],
                                 'fiber_num.npy'), sc_matrix)
            silent_remove(os.path.join(aging_data_dir,
                                       dst_dir[i],
                                       'fiber_num.mat'))



    
    
    