#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:05:15 2018

@author: asier
"""

import os
import numpy  as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

import aging as ag
os.chdir(os.path.join(ag.__path__[0], 'classifier'))


age = pd.read_csv('age.csv', header=None, names=['age'])
id_age = pd.read_csv('subjid.csv', header=None, names=['id'])

age_data = pd.concat([age, id_age], axis=1)


for idx, element in enumerate(age_data['id']):
    if element[0] == 'Y':
        age_data['id'][idx] = 'young_' + element[1:-1]
    elif element[0] == 'O':
        age_data['id'][idx] = 'old_' + element[1:-1]
    else:
        age_data['id'][idx] = element[:-1]
    
age_data = age_data.set_index('id')   

motion_data = pd.read_excel('motion_parameters_all.xls')
for idx, element in enumerate(motion_data['Code']):
    motion_data['Code'][idx] = str(element)
    
motion_data = motion_data.set_index('Code')   


full_data = age_data.merge(motion_data, how='outer', right_index=True, left_index=True)
full_data = full_data.groupby(full_data.index).sum()


for idx, element in enumerate(full_data['age']):
    if np.isnan(element):
        print(full_data.index[idx])


x, y = full_data["age"], full_data[" Mean_FD_Power"]
ax = sns.regplot(x=x, y=y, marker="+")


sns.jointplot(x=x, y=y)




