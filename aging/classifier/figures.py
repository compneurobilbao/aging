#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:28:42 2016

@author: asier
"""

## FIGURES

import numpy  as np
import seaborn as sns
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

###############
# PANEL 1 - MLE corr and MLE corr shuf


MLE = np.squeeze(np.load('corVsDesc_MLE.npy'))
MLE_shuf = np.squeeze(np.load('corVsDesc_MLE_shuf.npy'))
sigma_shuf = np.squeeze(np.load('sigma_MLE_shuf.npy'))

x = np.linspace(0, 200, 200)
plt.plot(x, MLE, label='MLE')

plt.hold(True)

plt.plot(x, MLE_shuf, color='#CC4F1B', label='MLE shuffled')
plt.fill_between(x, MLE_shuf-sigma_shuf, MLE_shuf+sigma_shuf, alpha=0.5, 
                 edgecolor='#CC4F1B', facecolor='#FF9848') 

plt.axis([0, 200, 0, 1])
plt.xlabel('Number of Descriptors')
plt.ylabel('corr(MLE, age)')
plt.legend()
plt.savefig('/home/asier/Desktop/AGING/motion_correction/figures/fig5/panel1.eps', format='eps', dpi=1000)




###############
# PANEL 4- MAE and MAE shuf

ordered_data = np.load('ordered_data_ext_int_mae.npy')
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


plt.axis([0, 100, 0, 30])
plt.xlabel('Number of Descriptors')
plt.ylabel('Cross-Validated MAE')
plt.legend()
plt.savefig('/home/asier/Desktop/AGING/motion_correction/figures/fig5/panel4.eps', format='eps', dpi=1000)


###############
# PANEL 5- prediction MAE ## ASK WHAT TO DO

print(np.argmin(results[1:114]))# 25 descriptors
print(np.min(results[1:114])) # 6.96142488891
min_error = np.min(results[1:114])


number_of_descriptors = np.argmin(results[1:114]) + 1 # +1 !!!
y = pd.read_csv('age.csv', header=None)
X = ordered_data[:,:number_of_descriptors]

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    m = metrics.mean_absolute_error(y_test, y_pred)
    print(m)
    if (m < min_error+0.1) and (m > min_error-0.1): break

import scipy
print(scipy.stats.pearsonr(y_test, y_pred))

plt.scatter(y_pred,y_test)
plt.plot(range(100),range(100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.savefig('/home/asier/Desktop/AGING/motion_correction/figures/fig5/panel5.eps', format='eps', dpi=1000)
 

################# data migration
import scipy.io as sio
sio.savemat('ordered_data.mat', {'ordered_desc':ordered_data})

###############
# PANEL 6- prediction MAE ## ASK WHAT TO DO

from nilearn import image
from nilearn import plotting as plt
import matplotlib.pyplot
import nibabel as nib
import os
import numpy as np
import aging as ag


os.chdir(os.path.join(ag.__path__[0], 'classifier'))
statmap = nib.load('netMode_SF_intExt_participation.nii')
# First plot the map for the PCC: index 4 in the atlas
plt.plot_glass_brain(statmap, threshold=0, colorbar=True,cmap=matplotlib.pyplot.cm.autumn, display_mode='lyrz', vmax = 4, vmin = 0, output_file = 'panel6.png')
