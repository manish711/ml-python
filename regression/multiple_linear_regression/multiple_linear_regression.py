# -*- coding: utf-8 -*-
"""
@author: manishnarang
Multiple Linear Regression
"""
#Importing Libraries
import numpy as np #contains mathematical tools.
import matplotlib.pyplot as plt #to help plot nice charts.
import pandas as pd #to import data sets and manage data sets.

#Importing Data Set - difference between the independent variables and the dependent variables.
dataSet = pd.read_csv('50_Startups.csv')
X = dataSet.iloc[:, :-1].values
Y = dataSet.iloc[:, -1].values

#Encode Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Split the training set and test set - a test set on which we test the performance of this machine learning model and the performance on the test set shouldn't be that different from the performance on the training sets because this would mean that the machine learning models understood well the correlations and didn't learn them by heart.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

# Fitting Multiple Linear Regression to the training set - y = b + m0x0 + m1x1...
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set result 
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2) #This will display any numerical value with 2 decimals after comma
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)),1))