from __future__ import absolute_import, division, print_function
import os
from distutils.dir_util import copy_tree
import aging as ag
import scipy
import scipy.io as sio
import numpy as np

data_path = os.path.join(ag.__path__[0], 'data')
aging_data_dir = os.path.join(data_path, 'subjects')
container_data_dir = os.path.join(data_path, 'container_data')  # ID_subj,FCpil

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

    FC_matrix = np.empty((2514, 2514, len(ID_subj)))

    for i, idx in enumerate(ID_subj):
        print(idx)
        folder_path = os.path.join(aging_data_dir, idx)
        time_series = sio.loadmat(os.path.join(folder_path, 'time_series.mat'))
        fc = np.corrcoef(time_series['time_series'].T)
        # TODO: handle NaNs
        # if np.argwhere(np.isnan(fc)): print('isnan')
        FC_matrix[:, :, i] = fc
    np.save(os.path.join(container_data_dir, 'ID_subj'), ID_subj)
    pass


def generate_SC():
    ID_subj = np.load(os.path.join(container_data_dir, 'ID_subj.npy'))

    SC_matrix = np.empty((2514, 2514, len(ID_subj)))

    for i, idx in enumerate(ID_subj):
        print(idx)
        folder_path = os.path.join(aging_data_dir, idx)
        fiber_num = sio.loadmat(os.path.join(folder_path, 'fiber_num.mat'))
        sc = scipy.sparse.csr_matrix(fiber_num['fiber_num'])
        # TODO: handle NaNs
        # if np.argwhere(np.isnan(fc)): print('isnan')

        SC_matrix[:, :, i] = sc

    
    pass


def generate_data_containers():

    if not os.path.exists(container_data_dir):
        os.makedirs(container_data_dir)

    generate_age_sex_ID()

    generate_FC()
    generate_SC()
    pass
 
        
        
        
        
        
        