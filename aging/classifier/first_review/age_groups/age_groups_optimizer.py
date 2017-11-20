#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 08:59:05 2016

@author: asier
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

import aging as ag
os.chdir(os.path.join(ag.__path__[0], 'classifier',
                      'first_review',
                      'age_groups'))

"""
MAE
"""

# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('data_ext.csv', header=None).T
data_int = pd.read_csv('data_int.csv', header=None).T

data = pd.concat((data_ext, data_int), axis = 1, ignore_index = True)
# import y data, age
y = pd.read_csv('age.csv', header=None)
y = np.array(y)

### YOUNG: <25
idx_young = np.where(y < 25)[0]
y_young = np.take(y, idx_young)
X_young = data.iloc[idx_young]

ordered_data, best_idx_list, results_young, sigma = optimize(X_young,
                                                             y_young,
                                                             nexp=100)

np.save('best_idx_ext_int_mae_young.npy', best_idx_list)
np.save('ordered_data_ext_int_mae_young.npy',ordered_data)
np.save('results_ext_int_mae_young.npy', results_young)
np.save('sigma_ext_int_mae_young.npy', sigma)
#results_young = np.squeeze(np.load('results_ext_int_mae_young.npy'))


#plt.plot(results_young)
print(np.argmin(results_young[1:60]))
print(np.min(results_young[1:60]))


### ADULT: 25<X<60
idx_adult = np.where((y > 25) & (y < 60))[0]
y_adult = np.take(y, idx_adult)
X_adult = data.iloc[idx_adult]

ordered_data, best_idx_list, results_adult, sigma = optimize(X_adult,
                                                             y_adult,
                                                             nexp=100)

np.save('best_idx_ext_int_mae_adult.npy', best_idx_list)
np.save('ordered_data_ext_int_mae_adult.npy',ordered_data)
np.save('results_ext_int_mae_adult.npy', results_adult)
np.save('sigma_ext_int_mae_adult.npy', sigma)
#results_adult = np.squeeze(np.load('results_ext_int_mae_adult.npy'))

#plt.plot(results_adult)
print(np.argmin(results_adult[1:60]))
print(np.min(results_adult[1:60]))


### OLD: 25<X<60
idx_old = np.where(y > 60)[0]
y_old = np.take(y, idx_old)
X_old = data.iloc[idx_old]

ordered_data, best_idx_list, results_old, sigma = optimize(X_old,
                                                           y_old,
                                                           nexp=100)

np.save('best_idx_ext_int_mae_old.npy', best_idx_list)
np.save('ordered_data_ext_int_mae_old.npy',ordered_data)
np.save('results_ext_int_mae_old.npy', results_old)
np.save('sigma_ext_int_mae_old.npy', sigma)
#results_old = np.squeeze(np.load('results_ext_int_mae_old.npy'))

#plt.plot(results_old)
print(np.argmin(results_old[1:60]))
print(np.min(results_old[1:60]))


def optimize(data, y, nexp=10):
    idx_set = set(range(200))
    best_idx_list = np.zeros([200,1], dtype='int')
    results = np.zeros([200,1])
    sigma = np.zeros([200,1])
    
    # First element
    ordered_data = data.loc[:,0]
    idx_set.remove(0)
    
    
    # MAE as a function of descriptor number
    lm = LinearRegression()
    for i in range(1, 60):
    
        for idx in idx_set:
    
            new_column = data.loc[:, idx]
            X = pd.concat((ordered_data, new_column), axis=1, ignore_index=True)
    
            r2_val = np.zeros([nexp, 1])
    
            for j in range(nexp):
                # Create from _test another _val set
                X_train, X_test, y_train, y_test = train_test_split(X,
                                                                    y)
    
                lm.fit(X_train, y_train)
                y_pred = lm.predict(X_test)
                r2_val[j] = metrics.mean_absolute_error(y_test, y_pred)
    
            if np.mean(r2_val) < results[i] or results[i] == 0:
                results[i] = np.mean(r2_val)
                sigma[i] = np.std(r2_val)
                best_descriptor = new_column
                best_idx = idx
            elif np.mean(r2_val) < 0:
                print("error in r2 value")
        
        print("loop", i, "best", best_idx)
        idx_set.remove(best_idx) 
        best_idx_list[i] = int(best_idx)    
        ordered_data = pd.concat((ordered_data,best_descriptor),axis=1, ignore_index=True)
        
    return ordered_data, best_idx_list, results, sigma