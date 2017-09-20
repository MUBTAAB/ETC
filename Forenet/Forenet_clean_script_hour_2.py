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
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

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

ilist = []
for i in clist:
    if ('date' in i or 
#        'month' in i or 
#        'hour' in i or 
#        'dayofweek' in i or 
#        ('v02' in i and int(i[-1]) <= 3) or 
#        ('v03' in i and int(i[-1]) <= 5) or
#        ('v04' in i and int(i[-1]) <= 3) or 
#        ('v05'in i and int(i[-1]) <= 3) or 
#        ('v06' in i and int(i[-1]) <= 5)  or
        ('v07_7' in i)  or
        'v01_close_mean_lag_back0' in i):
        ilist.append(i)

clist = ilist

dataset3['v01_close_mean_lag_back0'] = dataset3['v01_close_mean_lag_back0'] > dataset3['v01_close_mean_lag_back2']

mindate = '20040101'
splitdate = '20051001'

dataset_validation = dataset3[dataset3.date >= dt.datetime.strptime(splitdate,'%Y%m%d').date()]
dataset3 = dataset3[dataset3.date >= dt.datetime.strptime(mindate,'%Y%m%d').date()]
dataset3 = dataset3[dataset3.date < dt.datetime.strptime(splitdate,'%Y%m%d').date()]


dataset3 = dataset3[clist]
dataset_validation = dataset_validation[clist]


dataset4 = dataset3[filter_list(clist,['date'])]



X = dataset4[filter_list(list(dataset4.columns),['v01_close_mean_lag_back0'])].values
y = dataset4['v01_close_mean_lag_back0'].values
#del dataset4
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

forenet = Sequential()
# Adding the input layer and the first hidden layer
forenet.add(Dense(output_dim = 300, init = 'uniform', activation = 'sigmoid', input_dim = X_train.shape[1]))
#forenet.add(Dropout(0.3))
forenet.add(Dense(output_dim = 200, init = 'uniform', activation = 'sigmoid'))
#forenet.add(Dropout(0.2))
forenet.add(Dense(output_dim = 100, init = 'uniform', activation = 'sigmoid'))
#forenet.add(Dropout(0.1))
#forenet.add(Dropout(0.25))
forenet.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



forenet.compile(optimizer = 'Nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
forenet.fit(X_train, y_train, batch_size = 1, nb_epoch = 100, verbose = 1)


# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = forenet.predict(X_test)
dfresult = pd.DataFrame({'fact':y_test.flatten(), 'expected':y_pred.flatten()})
dfresult['fact'].plot()
dfresult['expected'].plot()
plt.xlabel('fact')
plt.ylabel('expected')
plt.show()


dfresult['Hit'] = (dfresult['expected'] > 0.5) == dfresult['fact']
print('hitrate test: ' + str(np.mean(dfresult['Hit'])))

corr = stats.pearsonr(dfresult['fact'], dfresult['expected'])

del corr, y_pred

totalpred = forenet.predict(X)
dataset3['expected'] = totalpred
#del X, y,totalpred, X_train, X_test, y_train, y_test

plt.plot(dataset3['date'].values,dataset3['v01_close_mean_lag_back0'].values)
plt.plot(dataset3['date'].values,dataset3['expected'].values)
plt.gcf().autofmt_xdate()
plt.show()


from sklearn.metrics import confusion_matrix

dataset3['motion_expected'] = dataset3['expected'] > 0.5 #np.mean(dataset3['expected'].values)
dataset3['true_motion'] = dataset3['v01_close_mean_lag_back0']
cm = confusion_matrix(dataset3['true_motion'].values, dataset3['motion_expected'].values)
print('hitrate train: '+ str(np.mean(dataset3['true_motion'] == dataset3['motion_expected'])))

dataset_validation2 = dataset_validation[filter_list(clist,['date'])]

X_valid = dataset_validation2[filter_list(list(dataset_validation2.columns),['v01_close_mean_lag_back0'])].values
X_valid = sc.fit_transform(X_valid)
y_valid = dataset_validation2['v01_close_mean_lag_back0'].values

totalpred_valid = forenet.predict(X_valid)
dataset_validation['expected'] = totalpred_valid

plt.plot(dataset_validation['date'].values,dataset_validation['v01_close_mean_lag_back0'].values)
plt.plot(dataset_validation['date'].values,dataset_validation['expected'].values)
plt.gcf().autofmt_xdate()
plt.show()

#del X_valid, y_valid, totalpred_valid, dataset_validation2


dataset_validation['motion_expected'] = dataset_validation['expected'] > 0.5 #np.mean(dataset_validation['expected'].values)
dataset_validation['true_motion'] = dataset_validation['v01_close_mean_lag_back0']

dataset_validation = dataset_validation[['date','expected','true_motion','motion_expected']]
print('hitrate validation: '+ str(np.mean(dataset_validation['true_motion']==dataset_validation['motion_expected'])))

cm_validation = confusion_matrix(dataset_validation['true_motion'].values, dataset_validation['motion_expected'].values)


