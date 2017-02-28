#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:19:04 2017

@author: asier
"""
import os
import numpy  as np
import seaborn as sns
import pandas as pd
import scipy.io as sio

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
os.chdir(os.path.join(ag.__path__[0], 'classifier'))


ordered_data = np.load('ordered_data_ext_int_mae.npy')
results = np.squeeze(np.load('results_ext_int_mae.npy'))
sigma = np.squeeze(np.load('sigma_ext_int_mae.npy'))


print(np.argmin(results[1:114])) # 
print(np.min(results[1:114])) # 

number_of_descriptors = np.argmin(results[1:114])+1 #WARNING! is 1:, so +1
y = np.array(pd.read_csv('age.csv', header=None))
X = ordered_data[:,:number_of_descriptors]
error = np.zeros((y.shape[0]))


lm = LinearRegression()
lm.fit(X, y)
for i in range(y.shape[0]):
    y_test = y[i]
    X_test = X[i,:]
    
    y_pred = lm.predict(X_test)
    error[i] = y_test - y_pred

plt.scatter(y, error)
plt.ylabel("prediction error")
plt.xlabel("age")

import scipy
print(scipy.stats.pearsonr(y[:,0], error))                    

import scipy.io as sio
sio.savemat('error_full_classifier.mat',{'error':error})   