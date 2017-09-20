# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:16:12 2017

@author: Mucs_b
"""

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
from scipy import stats
from sklearn.linear_model import LinearRegression
"""
from keras.models import Sequential
from keras.layers import Dense
"""
import warnings
warnings.filterwarnings("ignore")

print('A')
#import only first 100 rows
path = 'C:/Users/mucs_b/Desktop/Python projects/Local forenet/'
dataset = pd.read_csv(path+'EURUSD.txt',nrows = 1500000)

#rename columns
dataset.columns = ['ticker', 'data_date', 'data_time', 'open', 'high', 'low', 'close', 'vol']

#delete colunms
del dataset['ticker']
del dataset['vol']

#create new colunms based on fast calculations
dataset['fluct'] = ((dataset['high']-dataset['low'])/((dataset['high']+dataset['low'])/2))*100


#complex calculations based on iterations create list then add list to dataset
year = []
month = []
day = []

for i in dataset['data_date']:
   y = str(i)[:4]
   m = str(i)[4:-2]
   d = str(i)[-2:]
   year.append(y) 
   month.append(m) 
   day.append(d) 
   
dataset['year']= year[:]
dataset['month']= month[:]
dataset['day']= day[:]

del year,month,day,i,y,m,d


print('B')
#restructuring, pivoting datasets, aggregation
dataset3 = pd.pivot_table(dataset, values=['close', 'open', 'fluct'], index=['year', 'month', 'day'], aggfunc=[np.mean,np.max])

#getting rid of multiindex then renaming the colunms
dataset3.reset_index(inplace=True) 
dataset3.columns=['year', 'month', 'day','close_mean','close_max','open_mean','open_max','fluct_mean','fluct_max']

del dataset

#iterating function for multiple lag variables
for no in range(10):
    lag = []
    i = 0
    while len(lag) < len(dataset3):
        if i <= no:
            x = np.nan
            lag.append(x)
        else: 
            x = dataset3['close_mean'].values[i-no]
            lag.append(x)
        i += 1
        if len(lag) == len(dataset3):    
            dataset3['v01_close_mean_lag_back'+str(no)]= lag

del i, lag, no, x

dataset3 = dataset3.drop(['v01_close_mean_lag_back1'],1)
dataset3 = dataset3[dataset3.v01_close_mean_lag_back9 > 0]
#dataset3 = dataset3.head(n=4)
dataset4 = dataset3[:]


print('C')
for z in range(1,10):
    #print('z =' + str(z))
    lPred = []
    lCoeff = []
    for i in range(len(dataset4)):
        #print('i =' + str(i))
        #print('fact = ' + str(dataset4['close_mean'].values[i]))
        Y_values = dataset4.iloc[i, 10:11+z].values
        #print('Yvalues = ' + str(Y_values))
        X_values = [i+1 for i in range(len(Y_values))][::-1]
        regressor_1 = LinearRegression()
        regressor_1.fit(np.transpose(np.matrix(X_values)), np.transpose(np.matrix(Y_values)))
        pred = regressor_1.predict(len(Y_values)+2)
        pred = pred[0][0]
        pred2 = pred > Y_values[0]
        #print('Y0, pred, pred2'+ str([Y_values[0],pred,pred2]))#,dataset3['close_mean'].values[i] > dataset3['v01_close_mean_lag_back2'].values[i]]))
        coeff = regressor_1.coef_[0][0]
        lPred.append(pred2)
        lCoeff.append(coeff)
    dataset3['v02_linregcoef'+str(z)] = lCoeff
    dataset3['v03_linregpred'+str(z)] = lPred    
    
del z,i,Y_values,lPred,X_values,pred,lCoeff,coeff

cols = list(dataset3.columns[0:7])+sorted(dataset3.columns[7:])
dataset3 = dataset3[cols]

print('C.2')
for z in range(1,10):
    lPred = []
    lCoeff = []
    for i in range(len(dataset3)):
        Y_values = dataset4.iloc[i, 16:17+z].values
        X_values = [i for i in range(len(Y_values))][::-1]
        regressor_1 = LinearRegression()
        regressor_1.fit(np.transpose(np.matrix(X_values)), np.transpose(np.matrix(Y_values)))
        pred = regressor_1.predict(len(Y_values)+2)
        pred = pred[0][0]
        #pred = pred > Y_values[0] #dataset4['v01_close_mean_lag_back2'].values[i]
        coeff = regressor_1.coef_[0][0]
        lPred.append(pred)
        lCoeff.append(coeff)
    dataset3['v04_lincoefderiv'+str(z)] = lCoeff
    #dataset3['v04_linregpred'+str(z)] = lPred   
    
del z,i,lPred,Y_values,X_values,pred,lCoeff,coeff

print('D')
for z in range(1,10):
    #print('z =' + str(z))
    lPred = []
    lCoeff = []
    for i in range(len(dataset4)):
        #print('i =' + str(i))
        #print('fact = ' + str(dataset4['close_mean'].values[i]))
        Y_values = dataset4.iloc[i, 10:11+z].values
        #print(Y_values)
        X_values = [np.log(i+1) for i in range(len(Y_values))][::-1]
        regressor_1 = LinearRegression()
        regressor_1.fit(np.transpose(np.matrix(X_values)), np.transpose(np.matrix(Y_values)))
        pred = regressor_1.predict(np.log(len(Y_values)+2))
        pred = pred[0][0]
        pred2 = pred > Y_values[0] #dataset4['close_mean'].values[i]
        #print([Y_values[0],pred,pred2])
        coeff = regressor_1.coef_[0][0]
        lPred.append(pred2)
        lCoeff.append(coeff)
    dataset3['v05_logregcoef'+str(z)] = lCoeff
    dataset3['v06_logpred'+str(z)] = lPred

    
del z,i,lPred,Y_values,X_values,pred,lCoeff,coeff


print('D.1')
dataset3['date'] = [str(dataset3['year'].values[i])+str(dataset3['month'].values[i])+str(dataset3['day'].values[i]) for i in range(len(dataset3))]
import datetime as dt
dataset3['date']  = [dt.datetime.strptime(d,'%Y%m%d').date() for d in dataset3['date']]

cols = list(dataset3.columns)
cols = cols[-1:] + cols[:-1]
dataset3 = dataset3[cols]

dataset3['dayofweek'] = [d.weekday() for d in dataset3['date']]
dummydays = pd.get_dummies(dataset3['dayofweek'])
print('D.2')
dummydays.columns = ['dayofweek'+str(i) for i in list(sorted(set(dataset3['dayofweek'].values)))]
dataset3 = pd.concat([dataset3, dummydays], axis=1, join='inner')

print('F')
dummymonth = pd.get_dummies(dataset3['month'])
dummymonth.columns = ['month'+str(i) for i in list(sorted(set(dataset3['month'].values)))]
dataset3 = pd.concat([dataset3, dummymonth], axis=1, join='inner')

dataset3 = dataset3.drop(['dayofweek'],1)
dataset3 = dataset3.drop(['day'],1)
dataset3 = dataset3.drop(['month'],1)
dataset3 = dataset3.drop(['year'],1)
dataset3 = dataset3.drop(['dayofweek0'],1)
dataset3 = dataset3.drop(['month01'],1)


"""
print('v03_linregpred')
for i in range(1,10):
   plt.plot(dataset3['date'].values,dataset3['close_mean'].values)
   plt.plot(dataset3['date'].values,dataset3['v03_linregpred'+str(i)].values)
   plt.gcf().autofmt_xdate()
   plt.show()

print('')
print('v06_logpred')
for i in range(1,10):   
   plt.plot(dataset3['date'].values,dataset3['close_mean'].values)
   plt.plot(dataset3['date'].values,dataset3['v06_logpred'+str(i)].values)
   plt.gcf().autofmt_xdate()
   plt.show()
"""

print('v03_linregpred')
for i in range(1,10):
    print([i,np.average((dataset3['close_mean'] > dataset3['v01_close_mean_lag_back2']) == dataset3['v03_linregpred'+str(i)])])
print('')
print('v06_logpred')
for i in range(1,10):   
    print([i,np.average((dataset3['close_mean'] > dataset3['v01_close_mean_lag_back2']) == dataset3['v06_logpred'+str(i)])])


cols = list(dataset3.columns[0:7])+sorted(dataset3.columns[7:])
dataset3 = dataset3[cols]
print('G')


dataset3.to_csv('dataset3_day.csv', sep = ',', index = False)

