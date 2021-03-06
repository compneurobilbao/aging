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
                      'cuadratic_features'))

"""
MAE
"""

# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('data_ext.csv', header=None).T
data_int = pd.read_csv('data_int.csv', header=None).T

data = pd.concat((data_ext, data_int),
                 axis=1,
                 ignore_index=True)
# import y data, age
y = pd.read_csv('age.csv', header=None)
y = np.array(y)

ordered_data, best_idx_list, results, sigma = optimize(data,
                                                       y,
                                                       nexp=100)

np.save('best_idx_ext_int_mae_cuadratic.npy', best_idx_list)
np.save('ordered_data_ext_int_mae_cuadratic.npy', ordered_data)
np.save('results_ext_int_mae_cuadratic.npy', results)
np.save('sigma_ext_int_mae_cuadratic.npy', sigma)


def optimize(data, y, nexp=10):
    idx_set = set(range(200))
    best_idx_list = np.zeros([200, 1], dtype='int')
    results = np.zeros([200, 1])
    sigma = np.zeros([200, 1])

    # First element
    ordered_data = data.loc[:, 0]
    idx_set.remove(0)

    # MAE as a function of descriptor number
    lm = LinearRegression()
    for i in range(1, 101):
        for idx in idx_set:
            new_column = data.loc[:, idx]
            X = pd.concat((ordered_data, new_column),
                          axis=1,
                          ignore_index=True)

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
        ordered_data = pd.concat((ordered_data, best_descriptor),
                                 axis=1,
                                 ignore_index=True)

    return ordered_data, best_idx_list, results, sigma


results = np.load('results_ext_int_mae_cuadratic.npy')
# plt.plot(results)
print(np.argmin(results[1:59]))
print(np.min(results[1:59]))

"""
Results:

Cuadratic features: 36 features, 5.85 min mean error
"""


"""
Fitting the features
"""

# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('data_ext.csv', header=None).T
data_int = pd.read_csv('data_int.csv', header=None).T

data = pd.concat((data_ext, data_int),
                 axis=1,
                 ignore_index=True)
# import y data, age
y = pd.read_csv('age.csv', header=None)
y = np.array(y)

poly_data = np.zeros((155, 200))
for i in range(200):
    poly = np.ndarray.flatten(np.polyfit(data.loc[:, i], y, deg=2))
    poly_data[:, i] = np.polyval(poly, data.loc[:, i])

data = pd.DataFrame(poly_data)
ordered_data, best_idx_list, results, sigma = optimize(data,
                                                       y,
                                                       nexp=100)

np.save('best_idx_ext_int_mae_cuadratic_polyfit.npy', best_idx_list)
np.save('ordered_data_ext_int_mae_cuadratic_polyfit.npy', ordered_data)
np.save('results_ext_int_mae_cuadratic_polyfit.npy', results)
np.save('sigma_ext_int_mae_cuadratic_polyfit.npy', sigma)


results = np.load('results_ext_int_mae_cuadratic_polyfit.npy')
# plt.plot(results)
print(np.argmin(results[1:59]))
print(np.min(results[1:59]))

"""
Results:

Cuadratic features and fitting: 57 features, 5.87 min mean error
"""

# PLOT


ordered_data = np.load('ordered_data_ext_int_mae_cuadratic_polyfit.npy')
results = np.squeeze(np.load('results_ext_int_mae_cuadratic_polyfit.npy'))
sigma = np.squeeze(np.load('sigma_ext_int_mae_cuadratic_polyfit.npy'))


x = np.linspace(0, 200, 200)
plt.plot(x, results, label='MAE')
plt.fill_between(x, results-sigma, results+sigma, alpha=0.5) 


plt.axis([0, 100, 0, 30])
plt.xlabel('Number of Descriptors')
plt.ylabel('Cross-Validated MAE')
plt.legend()
plt.savefig('/home/asier/Desktop/cuadratic.eps', format='eps', dpi=1000)



# superpose plot
from os.path import join as opj
os.chdir(os.path.join(ag.__path__[0], 'classifier'))

cuadratic_path = os.path.join(ag.__path__[0], 'classifier',
                              'first_review',
                              'cuadratic_features')
    
   
results_c = np.squeeze(np.load(opj(cuadratic_path, 'results_ext_int_mae_cuadratic_polyfit.npy')))
sigma_c = np.squeeze(np.load(opj(cuadratic_path,'sigma_ext_int_mae_cuadratic_polyfit.npy')))

results = np.squeeze(np.load('results_ext_int_mae.npy'))
sigma = np.squeeze(np.load('sigma_ext_int_mae.npy'))

results_shuf = np.squeeze(np.load('results_ext_int_shuf_mae.npy'))
sigma_shuf = np.squeeze(np.load('sigma_ext_int_shuf_mae.npy'))

x = np.linspace(0, 200, 200)
plt.plot(x, results, label='MAE')
plt.fill_between(x, results-sigma, results+sigma, alpha=0.5) 

plt.hold(True)

plt.plot(x, results_shuf, color='#CC4F1B', label='MAE shuffled')
plt.fill_between(x, results_shuf-sigma_shuf, results_shuf+sigma_shuf, alpha=0.5, 
                 edgecolor='#CC4F1B', facecolor='#FF9848') 

plt.hold(True)

plt.plot(x, results_c, color='#C94F1B', label='MAE shuffled')
plt.fill_between(x, results_c-sigma_c, results_c+sigma_c, alpha=0.5, 
                 edgecolor='#C94F1B', facecolor='#F99848') 

plt.axis([0, 100, 0, 30])
plt.xlabel('Number of Descriptors')
plt.ylabel('Cross-Validated MAE')
plt.legend()
plt.savefig('/home/asier/Desktop/lin_cuad_shuff.eps', format='eps', dpi=1000)

