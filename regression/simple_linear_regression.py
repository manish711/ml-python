# -*- coding: utf-8 -*-
"""
@author: manishnarang
Simple Linear Regression
"""
#Importing Libraries
import numpy as np #contains mathematical tools.
import matplotlib.pyplot as plt #to help plot nice charts.
import pandas as pd #to import data sets and manage data sets.

#Importing Data Set - difference between the independent variables and the dependent variables.
dataSet = pd.read_csv('Salary_Data.csv')
X = dataSet.iloc[:, :-1].values
Y = dataSet.iloc[:, -1].values

#Split the training set and test set - a test set on which we test the performance of this machine learning model and the performance on the test set shouldn't be that different from the performance on the training sets because this would mean that the machine learning models understood well the correlations and didn't learn them by heart.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

# Fitting Simple Linear Regression to the training set - y = mx + b, where m is the slope (Gradient) {Change in y/ Change in x} and b is y-intercept (where the line crosses the Y axis)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set result -[ y-y1 = m(x-x1) ]
Y_pred = regressor.predict(X_test)

#Visualising the Training Set Results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Training Set Results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()