#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 08:59:05 2016

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


#####################
### FIGURES ###
#####################

##############
### CORR #####

# read CSV file directly from a URL and save the results
data_ext = np.array(pd.read_csv('data_ext.csv', header=None).T)
data_int = np.array(pd.read_csv('data_int.csv', header=None).T)

data = np.empty([155,200])
for i in range(0,199,2):
    data[:,i] = data_ext[:,i//2]
    data[:,i+1] = data_int[:,i//2]

# import y data, age
y = np.array(pd.read_csv('age.csv', header=None))

##
## MLE
##

corVsDesc = np.zeros(len(data.T));
for i in range(200):
    MLEdesc = np.column_stack((np.ones(len(y)), data[:,:i]));
    wml = np.dot(np.dot(np.linalg.pinv(np.dot(MLEdesc.T, MLEdesc)), MLEdesc.T) , y);
    # print np.corrcoef(y.T,np.dot(MLEdesc,wml).T)[0, 1]
    corVsDesc[i] = np.corrcoef(y.T,np.dot(MLEdesc,wml).T)[0, 1];

np.save('corVsDesc_MLE.npy', corVsDesc)


###################
### CORR shuf #####

# read CSV file directly from a URL and save the results
data_ext = np.array(pd.read_csv('data_ext.csv', header=None).T)
data_int = np.array(pd.read_csv('data_int.csv', header=None).T)

data = np.empty([155,200])
for i in range(0,199,2):
    data[:,i] = data_ext[:,i//2]
    data[:,i+1] = data_int[:,i//2]

# import y data, age
y = np.array(pd.read_csv('age.csv', header=None))
np.random.shuffle((y))

##
## MLE
##
Nexp = 100
corVsDesc = np.zeros(len(data.T))
sigma_corVsDesc = np.zeros(len(data.T))
results = np.zeros((200,Nexp))   


for j in range(Nexp):
    np.random.shuffle((y))
    print(j)

    for i in range(200):
        MLEdesc = np.column_stack((np.ones(len(y)), data[:,:i]));
        wml = np.dot(np.dot(np.linalg.pinv(np.dot(MLEdesc.T, MLEdesc)), MLEdesc.T) , y);
        #print np.corrcoef(y.T,np.dot(MLEdesc,wml).T)[0, 1]
        results[i,j] = np.corrcoef(y.T,np.dot(MLEdesc,wml).T)[0, 1];

results = np.nan_to_num(results)
corVsDesc = np.mean(results, axis = 1)
sigma_corVsDesc = np.std(results, axis = 1)
    

np.save('corVsDesc_MLE_shuf.npy', corVsDesc)
np.save('sigma_MLE_shuf.npy', sigma_corVsDesc)


##############
### MAE ######


# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('data_ext.csv', header=None).T
data_int = pd.read_csv('data_int.csv', header=None).T

data = pd.concat((data_ext, data_int), axis = 1, ignore_index = True)

# import y data, age
y = pd.read_csv('age.csv', header=None)
y = np.array(y)
##
## Optimizer
##

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
    

np.save('best_idx_ext_int_mae.npy', best_idx_list)
np.save('ordered_data_ext_int_mae.npy',ordered_data)
np.save('results_ext_int_mae.npy', results)
np.save('sigma_ext_int_mae.npy', sigma)



###############
### MAE Shuf ##





# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('data_ext.csv', header=None).T
data_int = pd.read_csv('data_int.csv', header=None).T

data = pd.concat((data_ext, data_int), axis = 1, ignore_index = True)

# import y data, age
y = pd.read_csv('age.csv', header=None)
y = np.array(y)
np.random.shuffle((y))
##
## Optimizer
##

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
    

np.save('best_idx_ext_int_shuf_mae.npy', best_idx_list)
np.save('ordered_data_ext_int_shuf_mae.npy',ordered_data)
np.save('results_ext_int_shuf_mae.npy', results)
np.save('sigma_ext_int_shuf_mae.npy', sigma)





