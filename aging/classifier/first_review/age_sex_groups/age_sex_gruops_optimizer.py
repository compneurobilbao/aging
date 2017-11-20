import os
import numpy  as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import scipy.io as sio

import matplotlib.pyplot as plt
os.chdir('/home/asier/Desktop/AGING/motion_correction')

###########
###########
###########
########### EXPERIMENT: Only MALE or FEMALE
## ONLY MALE

# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('male_ext.csv', header=None).T
data_int = pd.read_csv('male_int.csv', header=None).T

data = pd.concat((data_ext, data_int), axis = 1, ignore_index = True)
# import y data, age
y = pd.read_csv('male_age.csv', header=None)
y = np.array(y)
##
## Optimizer
##
desc_num = data.shape[1]

idx_set = set(range(desc_num))
best_idx_list = np.zeros(desc_num)
results = np.zeros(desc_num)
sigma = np.zeros(desc_num)
Nexp = 20

# First element
ordered_data = data.loc[:,0]
idx_set.remove(0)


# MAE as a function of descriptor number
lm = LinearRegression()
for i in range(1,95):
    
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
    

np.save('best_idx_male.npy', best_idx_list)
np.save('ordered_data_male.npy',ordered_data)
np.save('results_male.npy', results)
np.save('sigma_male.npy', sigma)



######
# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('female_ext.csv', header=None).T
data_int = pd.read_csv('female_int.csv', header=None).T

data = pd.concat((data_ext, data_int), axis = 1, ignore_index = True)

# import y data, age
y = pd.read_csv('female_age.csv', header=None)
y = np.array(y)
##
## Optimizer
##

idx_set = set(range(184))
best_idx_list = np.zeros([184,1])
results = np.zeros([184,1])
sigma = np.zeros([184,1])
Nexp = 20

# First element
ordered_data = data.loc[:,0]
idx_set.remove(0)


# MAE as a function of descriptor number
lm = LinearRegression()
for i in range(1,95):
    
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
    

np.save('best_idx_female.npy', best_idx_list)
np.save('ordered_data_female.npy',ordered_data)
np.save('results_female.npy', results)
np.save('sigma_female.npy', sigma)




###############
# PANEL 5- prediction MAE ## ASK WHAT TO DO

best_idx = np.load('best_idx_male.npy')
ordered_data = np.load('ordered_data_female.npy')
results = np.squeeze(np.load('results_female.npy'))
sigma = np.squeeze(np.load('sigma_female.npy'))


print(np.argmin(results[1:82])) # 25 descriptors
print(np.min(results[1:82])) # 6.96142488891

x = np.linspace(0, 153, 153)
plt.plot(x, results, label='MAE')
plt.fill_between(x, results[:,0]-sigma[:,0], results[:,0]+sigma[:,0], alpha=0.5) 



number_of_descriptors = np.argmin(results[1:95])
y = pd.read_csv('female_age.csv', header=None)
X = ordered_data[:,:number_of_descriptors]

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    m = metrics.mean_absolute_error(y_test, y_pred)
    print(m)
    if (m < np.min(results[1:])+0.1) and (m > np.min(results[1:])-0.1): break

import scipy
print(scipy.stats.pearsonr(y_test, y_pred))

plt.scatter(y_pred,y_test)
plt.plot(range(100),range(100))
plt.xlabel("predicted age")
plt.ylabel("real age")