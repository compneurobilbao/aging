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
                      'age_sex_groups'))

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
best_idx_list = np.zeros([200,1])
results = np.zeros([200,1])
sigma = np.zeros([200,1])
Nexp = 100

# First element
ordered_data = data.loc[:,0]
idx_set.remove(0)


# MAE as a function of descriptor number
lm = LinearRegression()
for i in range(1,116):
    
    for idx in idx_set:

        new_column = data.loc[:,idx]
        X = pd.concat((ordered_data,new_column),axis=1, ignore_index=True)

        r2_val = np.zeros([Nexp,1])
        
        for j in range(Nexp):
            X_train, X_test, y_train, y_test = train_test_split(X, y)
            
            # Create from _test another _val set
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.4,
                                                                random_state=1)
            X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                            y_test,
                                                            test_size=0.5,
                                                            random_state=1)

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
    best_idx_list[i] = best_idx    
    ordered_data = pd.concat((ordered_data,best_descriptor),axis=1, ignore_index=True)
    

#np.save('best_idx_ext_int_mae.npy', best_idx_list)
#np.save('ordered_data_ext_int_mae.npy',ordered_data)
#np.save('results_ext_int_mae.npy', results)
#np.save('sigma_ext_int_mae.npy', sigma)


#ordered_data = np.load('ordered_data_ext_int_mae.npy')
#results = np.squeeze(np.load('results_ext_int_mae.npy'))
#sigma = np.squeeze(np.load('sigma_ext_int_mae.npy'))


print(np.argmin(results[1:114])) # 
print(np.min(results[1:114])) # 
   
number_of_descriptors = np.argmin(results[1:114])+1 #WARNING! is 1:, so +1
y = np.array(pd.read_csv('age.csv', header=None))
X = ordered_data[:,:number_of_descriptors]
error = np.zeros((y.shape[0]))

for i in range(X.shape[0]):

    y_test = y[i]
    y_train = np.delete(y, (i), axis = 0)
    X_test = X[i,:]
    X_train = np.delete(X, (i), axis = 0)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_val)
    error[i] = y_val - y_pred
 
plt.scatter(y, error)
plt.ylabel("prediction error")
plt.xlabel("age")




