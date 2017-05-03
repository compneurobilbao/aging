#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:12:29 2017

@author: asier
"""


###########
###########
###########
########### EXPERIMENT: Only SC or only FC

# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('intFC.csv', header=None).T
data_int = pd.read_csv('extFC.csv', header=None).T

data = pd.concat((data_ext, data_int), axis = 1, ignore_index = True)

# import y data, age
y = pd.read_csv('age.csv', header=None)
y = np.array(y)
np.random.shuffle((y))
##
## Optimizer
##

idx_set = set(range(95))
best_idx_list = np.zeros([95,1])
results = np.zeros([95,1])
sigma = np.zeros([95,1])
Nexp = 95

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
    

np.save('best_idx_just_FC_mae.npy', best_idx_list)
np.save('ordered_data_just_FC_mae.npy',ordered_data)
np.save('results_just_FC_mae.npy', results)
np.save('sigma_just_FC_mae.npy', sigma)



######
# read CSV file directly from a URL and save the results
data_ext = pd.read_csv('intSC.csv', header=None).T
data_int = pd.read_csv('extSC.csv', header=None).T

data = pd.concat((data_ext, data_int), axis = 1, ignore_index = True)

# import y data, age
y = pd.read_csv('age.csv', header=None)
y = np.array(y)
np.random.shuffle((y))
##
## Optimizer
##

idx_set = set(range(95))
best_idx_list = np.zeros([95,1])
results = np.zeros([95,1])
sigma = np.zeros([95,1])
Nexp = 100

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
    

np.save('best_idx_just_SC_mae.npy', best_idx_list)
np.save('ordered_data_just_SC_mae.npy',ordered_data)
np.save('results_just_SC_mae.npy', results)
np.save('sigma_just_SC_mae.npy', sigma)




###############
# PANEL 5- prediction MAE ## ASK WHAT TO DO

ordered_data = np.load('ordered_data_just_FC_mae.npy')
results = np.squeeze(np.load('results_just_FC_mae.npy'))
sigma = np.squeeze(np.load('sigma_just_FC_mae.npy'))


print(np.argmin(results[1:114])) # 25 descriptors
print(np.min(results[1:114])) # 6.96142488891

x = np.linspace(0, 95, 95)
plt.plot(x, results, label='MAE')
plt.fill_between(x, results-sigma, results+sigma, alpha=0.5) 


number_of_descriptors = np.argmin(results[1:95])
y = pd.read_csv('age.csv', header=None)
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
plt.savefig('/home/asier/Desktop/AGING/motion_correction/figures/fig5/panel5.eps', format='eps', dpi=1000)
 

