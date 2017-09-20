# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:50:15 2017

@author: Mucs_b
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:20:44 2016

@author: mucs_b
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import grid_search

import warnings
warnings.filterwarnings("ignore")

path = 'C:/Users/mucs_b/Desktop/Python projects/Local forenet/'
dataset3 = pd.read_csv(path+'dataset3_hour.csv')
dataset_sample = dataset3.head()

dataset3['date']  = [dt.datetime.strptime(d,'%Y-%m-%d %H:00:00') for d in dataset3['date']]
clist = list(dataset3.columns)
a0_clist = clist[:]
def filter_list(full_list, excludes):
    s = set(excludes)
    return ([x for x in full_list if x not in s])

dataset_sample = dataset3.head()


month = False
hour = False
dayofweek = False
v01 = list(range(1))
v02 = list(range(4))
v03 = list(range(4))
v04 = list(range(4))
v05 = list(range(4))
v06 = list(range(4))
v07_1 = False
v07_2 = False
v07_3 = False
v07_4 = False
v07_5 = False
v07_6 = False
v07_7 = False
v07_8 = False
v07_9 = False

mindate = '20040101'
splitdate = '20051001'

n_estimators = 1
max_depth = 3
min_samples_leaf = 1
fold = 5

print(dt.datetime.now())
dVars = {'month':month, 'hour':hour, 'dayofweek':dayofweek, 'v01':[np.min(v01),np.max(v01)], 'v02':[np.min(v02),np.max(v02)], 'v03':[np.min(v03),np.max(v03)],'v04':[np.min(v04),np.max(v04)],'v05':[np.min(v05),np.max(v05)], 'v06':[np.min(v06),np.max(v06)], 'v07_1':v07_1, 'v07_2':v07_2, 'v07_3':v07_3, 'v07_4':v07_4, 'v07_5':v07_5, 'v07_6':v07_6, 'v07_7':v07_7, 'v07_9':v07_8, 'v07_9':v07_9}
dSample = {'mindate':mindate, 'splitdate':splitdate, 'fold':fold}
#lParam = {'n_estimators':n_estimators, 'max_depth':max_depth, 'min_samples_leaf':min_samples_leaf, 'fold':fold}

print(dVars)
print(dSample)
#print(lParam)

ilist = []
for i in clist:
    if ('date' in i or 
        ('month' in i and month == True)  or 
        ('hour' in i and hour == True) or 
        ('dayofweek' in i and dayofweek == True) or 
        ('v01' in i and int(i[-1]) in v01) or 
        ('v02' in i and int(i[-1]) in v02) or 
        ('v03' in i and int(i[-1]) in v03) or
        ('v04' in i and int(i[-1]) in v04) or 
        ('v05'in i and int(i[-1]) in v05) or 
        ('v06' in i and int(i[-1]) in v06)  or
        ('v07_1' in i and v07_1 == True)  or
        ('v07_2' in i and v07_2 == True)  or
        ('v07_3' in i and v07_3 == True)  or
        ('v07_4' in i and v07_4 == True)  or
        ('v07_5' in i and v07_5 == True)  or
        ('v07_6' in i and v07_6 == True)  or
        ('v07_7' in i and v07_7 == True)  or
        ('v07_8' in i and v07_8 == True)  or
        ('v07_9' in i and v07_9 == True)  or
        'v01_close_mean_lag_back0' in i):
        ilist.append(i)

clist = ilist
len(dataset3)
dataset3['v01_close_mean_lag_back0'] = dataset3['v01_close_mean_lag_back0'] > dataset3['v01_close_mean_lag_back2']

dataset_validation = dataset3[dataset3.date >= dt.datetime.strptime(splitdate,'%Y%m%d').date()]
dataset3 = dataset3[dataset3.date >= dt.datetime.strptime(mindate,'%Y%m%d').date()]
dataset3 = dataset3[dataset3.date < dt.datetime.strptime(splitdate,'%Y%m%d').date()]


dataset3 = dataset3[clist]
dataset_validation = dataset_validation[clist]


dataset4 = dataset3[filter_list(clist,['date'])]



X = dataset4[filter_list(list(dataset4.columns),['v01_close_mean_lag_back0'])].values
y = dataset4['v01_close_mean_lag_back0'].values
#del dataset4
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

"""
fore_rf = RandomForestClassifier(n_estimators = 10**n_estimators, max_depth = max_depth, min_samples_leaf = min_samples_leaf, verbose = False)
fore_rf.fit(X_train, y_train)

if fold > 0:
    print('cv: ' + str(np.mean(cross_val_score(fore_rf, X_train, y_train, cv=fold))))
else: 
    print('no cv scores')
"""


param_grid = {
                 'n_estimators': [10**1,10**4],
                 'max_depth': [2, 4, 10**5],
                 'min_samples_leaf': [1,100,1000]
             }

clf = RandomForestClassifier()
grid_clf = grid_search.GridSearchCV(clf, param_grid, cv=fold, verbose = 5)
grid_clf.fit(X, y)

print('best params:' + str(grid_clf.best_params_))
print('best params:' + str(grid_clf.best_score_))

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results

totalpred = grid_clf.predict(X)
dataset3['expected'] = totalpred
#del X, y,totalpred, X_train, X_test, y_train, y_test

from sklearn.metrics import confusion_matrix

dataset3['motion_expected'] = dataset3['expected'] > 0.5 #np.mean(dataset3['expected'].values)
dataset3['true_motion'] = dataset3['v01_close_mean_lag_back0']
cm = confusion_matrix(dataset3['true_motion'].values, dataset3['motion_expected'].values)
print('hitrate train: '+ str(np.mean(dataset3['true_motion'] == dataset3['motion_expected'])))

dataset_validation2 = dataset_validation[filter_list(clist,['date'])]

X_valid = dataset_validation2[filter_list(list(dataset_validation2.columns),['v01_close_mean_lag_back0'])].values
#X_valid = sc.fit_transform(X_valid)
y_valid = dataset_validation2['v01_close_mean_lag_back0'].values

totalpred_valid = grid_clf.predict(X_valid)
dataset_validation['expected'] = totalpred_valid
#del X_valid, y_valid, totalpred_valid, dataset_validation2


dataset_validation['motion_expected'] = dataset_validation['expected'] > 0.5 #np.mean(dataset_validation['expected'].values)
dataset_validation['true_motion'] = dataset_validation['v01_close_mean_lag_back0']

dataset_validation = dataset_validation[['date','expected','true_motion','motion_expected']]
print('hitrate validation: '+ str(np.mean(dataset_validation['true_motion']==dataset_validation['motion_expected'])))

cm_validation = confusion_matrix(dataset_validation['true_motion'].values, dataset_validation['motion_expected'].values)

print(dt.datetime.now())


