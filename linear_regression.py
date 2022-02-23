# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:17:15 2021
Simple Linear regression

@author: MubaH_dev
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




#importing dataset
dataset = pd.read_csv('Salaries.csv')


#independent varialble
X = dataset.iloc[:,0].values
#dependent varialble
Y = dataset.iloc[:,1].values

        
 #Split data into test an train
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/3, random_state=0)



#Fitting Simple Linear Regression To Training Test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
regressor.fit(X_train, Y_train)

#Predicting Test Result
X_test = X_test.reshape(-1,1)
Y_pred =regressor.predict(X_test)

#Visualizing results: Training set
plt.scatter(X_train,Y_train)
plt.plot(X_train,regressor.predict(X_train), color='red')
plt.title('Salary Vs Experience(Training set Result)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()

#Visualizing results: Testing set
#plt.scatter(X_test,Y_test)
#plt.plot(X_train,regressor.predict(X_train), color='red')
#plt.title('Salary Vs Experience(Test set Result)')
#plt.xlabel('Years of Experience')
#plt.ylabel("Salary")
plt.show()



