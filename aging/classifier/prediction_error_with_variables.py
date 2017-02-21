#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:56:46 2017

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
os.chdir('/home/asier/Desktop/AGING/motion_correction')


ordered_data = np.load('ordered_data_ext_int_mae.npy')
results = np.squeeze(np.load('results_ext_int_mae.npy'))
sigma = np.squeeze(np.load('sigma_ext_int_mae.npy'))


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
    y_pred = lm.predict(X_test)
    error[i] = y_test - y_pred

    
    
    
plt.scatter(y, error)
plt.ylabel("prediction error")
plt.xlabel("age")



###################### other responses
import scipy


info_limited = pd.read_csv('Participants_info.csv')
info_ID = np.array(info_limited.ppID)
           
subj_id_155 = pd.read_csv('subjid.csv', header=None)
subj_id_155 = np.array(subj_id_155)
id_155 = [subj[0][:-1] for subj in subj_id_155]

## ipaqtotal
ipaqq = np.array(info_limited.IpaqTotal)
ipaqq = np.array([subj_id for i, subj_id in enumerate(ipaqq) if not np.isnan(info_limited.IpaqTotal[i])])
ID_ipaqq = [str(subj_id) for i, subj_id in enumerate(info_ID) if not np.isnan(info_limited.IpaqTotal[i])]

idx_155_ipaqq = [i for i, subj in enumerate(id_155) if subj in ID_ipaqq]           
idx_limited_ipaqq = [i for i, subj in enumerate(ID_ipaqq) if subj in id_155]           


plt.scatter(ipaqq[idx_limited_ipaqq], error[idx_155_ipaqq])
plt.ylabel("prediction error")
plt.xlabel("ipaqq")
print(scipy.stats.pearsonr(ipaqq[idx_limited_ipaqq], error[idx_155_ipaqq]))                    


# MeanBoth - motor skill
motor = np.array(info_limited.MeanBoth)
motor = np.array([subj_id for i, subj_id in enumerate(motor) if not np.isnan(info_limited.MeanBoth[i])])
ID_motor = [str(subj_id) for i, subj_id in enumerate(info_ID) if not np.isnan(info_limited.MeanBoth[i])]

idx_155_motor = [i for i, subj in enumerate(id_155) if subj in ID_motor]           
idx_limited_motor = [i for i, subj in enumerate(ID_motor) if subj in id_155] 

plt.scatter(motor[idx_limited_motor], error[idx_155_motor])
plt.ylabel("prediction error")
plt.xlabel("motor")
print(scipy.stats.pearsonr(motor[idx_limited_motor], error[idx_155_motor]))                    


# Beckscore -depression
beck = np.array(info_limited.Beckscore)
beck = np.array([subj_id for i, subj_id in enumerate(beck) if not np.isnan(info_limited.Beckscore[i])])
ID_beck = [str(subj_id) for i, subj_id in enumerate(info_ID) if not np.isnan(info_limited.Beckscore[i])]

idx_155_beck = [i for i, subj in enumerate(id_155) if subj in ID_beck]           
idx_limited_beck = [i for i, subj in enumerate(ID_beck) if subj in id_155]

plt.scatter(beck[idx_limited_beck], error[idx_155_beck])
plt.ylabel("prediction error")
plt.xlabel("beck")
print(scipy.stats.pearsonr(beck[idx_limited_beck], error[idx_155_beck]))                    



print(scipy.stats.pearsonr(ipaqq[idx_limited_ipaqq], y[idx_155_ipaqq,0] ))
print(scipy.stats.pearsonr(motor[idx_limited_motor], y[idx_155_motor,0] ))
print(scipy.stats.pearsonr(beck[idx_limited_beck], y[idx_155_beck,0] ))


# MoCa -MCI
moca = np.array(info_limited.MoCa)
moca = np.array([subj_id for i, subj_id in enumerate(moca) if not np.isnan(info_limited.MoCa[i])])
ID_moca = [str(subj_id) for i, subj_id in enumerate(info_ID) if not np.isnan(info_limited.MoCa[i])]

idx_155_moca = [i for i, subj in enumerate(id_155) if subj in ID_moca]           
idx_limited_moca = [i for i, subj in enumerate(ID_moca) if subj in id_155]

plt.scatter(moca[idx_limited_moca], error[idx_155_moca])
plt.ylabel("prediction error")
plt.xlabel("moca")
print(scipy.stats.pearsonr(moca[idx_limited_moca], error[idx_155_moca]))                    



print(scipy.stats.pearsonr(ipaqq[idx_limited_ipaqq], y[idx_155_ipaqq,0] ))
print(scipy.stats.pearsonr(motor[idx_limited_motor], y[idx_155_motor,0] ))
print(scipy.stats.pearsonr(beck[idx_limited_beck], y[idx_155_beck,0] ))



plt.scatter(info_limited.MoCa, info_limited.Age)





ipaqq = np.array(info_limited.IpaqTotal)
ipaqq = np.array([subj_id for i, subj_id in enumerate(ipaqq) if not np.isnan(info_limited.IpaqTotal[i]) and info_limited.Age[i] > 25])
ID_ipaqq = [str(subj_id) for i, subj_id in enumerate(info_ID) if not np.isnan(info_limited.IpaqTotal[i]) and info_limited.Age[i] > 25]

idx_155_ipaqq = [i for i, subj in enumerate(id_155) if subj in ID_ipaqq]           
idx_limited_ipaqq = [i for i, subj in enumerate(ID_ipaqq) if subj in id_155]           


plt.scatter(ipaqq[idx_limited_ipaqq], error[idx_155_ipaqq])
plt.ylabel("prediction error")
plt.xlabel("ipaqq")
print(scipy.stats.pearsonr(ipaqq[idx_limited_ipaqq], error[idx_155_ipaqq]))                    




motor = np.array(info_limited.MeanBoth)
motor = np.array([subj_id for i, subj_id in enumerate(motor) if not np.isnan(info_limited.MeanBoth[i]) and info_limited.Age[i] > 30])
ID_motor = [str(subj_id) for i, subj_id in enumerate(info_ID) if not np.isnan(info_limited.MeanBoth[i]) and info_limited.Age[i] > 30]

idx_155_motor = [i for i, subj in enumerate(id_155) if subj in ID_motor]           
idx_limited_motor = [i for i, subj in enumerate(ID_motor) if subj in id_155] 

plt.scatter(motor[idx_limited_motor], error[idx_155_motor])
plt.ylabel("prediction error")
plt.xlabel("motor")
print(scipy.stats.pearsonr(motor[idx_limited_motor], error[idx_155_motor]))                    



### pcorr
from scipy import stats # and import p_corr from aging.proc
p_corr(error[idx_155_ipaqq] , motor[idx_limited_ipaqq], y[idx_155_ipaqq,0])
p_corr(error[idx_155_motor] , motor[idx_limited_motor], y[idx_155_motor,0])
p_corr(error[idx_155_moca] , motor[idx_limited_moca], y[idx_155_moca,0])
p_corr(error[idx_155_beck] , motor[idx_limited_beck], y[idx_155_beck,0])