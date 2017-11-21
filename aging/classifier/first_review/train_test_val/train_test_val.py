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
                      'train_test_val'))

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


idx_set = set(range(200))
best_idx_list = np.zeros([200,1], dtype='int')
results = np.zeros([200,1])
sigma = np.zeros([200,1])
Nexp = 10

data_use, X_val, y_use, y_val = train_test_split(data.loc[:, :],
                                                 y,
                                                 test_size=0.1)

# First element
ordered_data = data_use.loc[:,0]
idx_set.remove(0)



# MAE as a function of descriptor number
lm = LinearRegression()
for i in range(1, 50):

    for idx in idx_set:

        new_column = data_use.loc[:, idx]
        X = pd.concat((ordered_data, new_column), axis=1, ignore_index=True)

        r2_val = np.zeros([Nexp, 1])

        for j in range(Nexp):
            # Create from _test another _val set
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y_use,
                                                                test_size=0.33)

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
    

np.save('best_idx_ext_int_mae_train_test_val.npy', best_idx_list)
np.save('ordered_data_ext_int_mae_train_test_val.npy',ordered_data)
np.save('results_ext_int_mae_train_test_val.npy', results)
np.save('sigma_ext_int_mae_train_test_val.npy', sigma)
np.save('y_ext_int_mae_train_test_val.npy', y_use)



#ordered_data = np.load('ordered_data_ext_int_mae_train_test_val.npy')
#results = np.squeeze(np.load('results_ext_int_mae_train_test_val.npy'))
#sigma = np.squeeze(np.load('sigma_ext_int_mae_train_test_val.npy'))
#best_idx_list = np.load('best_idx_ext_int_mae_train_test_val.npy')
# y_use = np.load('best_idx_ext_int_mae_train_test_val.npy')

plt.plot(results)
print(np.argmin(results[1:50]))
print(np.min(results[1:50]))

number_of_descriptors = np.argmin(results[1:114]) + 1  # WARNING! is 1:, so +1
y = y_use
X = np.array(ordered_data)[:, :number_of_descriptors]
error = np.zeros((y_val.shape[0]))
y_pred = np.zeros((y_val.shape[0]))


lm = LinearRegression()
lm.fit(X, y)

X_val_ = np.array(X_val)
# order
X_val_desc = np.take(X_val_, best_idx_list[:number_of_descriptors],
                     axis=1)[:, :, 0]

for i in range(y_val.shape[0]):
    X_ = X_val_desc[i, :]
    X_ = X_.reshape(1, -1)
    y_pred[i] = lm.predict(X_)
    error[i] = y_val[i] - y_pred[i]

#plt.scatter(y_val, error)
#plt.ylabel("prediction error")
#plt.xlabel("age")

plt.scatter(y_val, y_pred)
plt.ylabel("predicted")
plt.xlabel("real")

np.mean(abs(error))

import scipy
print(scipy.stats.pearsonr(y_val[:,0], y_pred))    

"""
Results:

Ran twice. This set up requires to be run many times to evaluate the effect
of having a new VALIDATION SET. As far as I've seen, the results are a bit worse
than with our previous set up. 6.5-7 years error. This error rate is quite dependant
on the % of the set extracted for validation and is just for a single run. 
"""