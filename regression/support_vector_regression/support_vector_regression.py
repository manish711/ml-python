# -*- coding: utf-8 -*-
"""
@author: manishnarang
Support Vector Regression
"""
#Importing Libraries
import numpy as np #contains mathematical tools.
import matplotlib.pyplot as plt #to help plot nice charts.
import pandas as pd #to import data sets and manage data sets.

#Importing Data Set - difference between the independent variables and the dependent variables.
dataSet = pd.read_csv('Position_Salaries.csv')
X = dataSet.iloc[:, 1:-1].values
Y = dataSet.iloc[:, -1].values

#In SVR, we have to apply feature scaling to both dependent and indepdent variable.
#Both should be same array type. Hence, convert Y into 2-D Array
Y = Y.reshape(len(Y), 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
#We don't have to use same object (sc) on Dependent variable because it is going to compute the mean and the standard deviation of X variable.
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

#Training the SVR Model on whole Dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') #Gaussian Radial Basis Function (most widely used) https://data-flair.training/blogs/svm-kernel-functions/
regressor.fit(X,Y)

#Predicting a new result based on experience level of 6.5
sc_Y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#Visualising the SVR Results
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_Y.inverse_transform(regressor.predict(X)), color='blue')
plt.title('Support Vector Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising the SVR Results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_X.inverse_transform(X), sc_Y.inverse_transform(Y), color='red')
plt.plot(X_grid, sc_Y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='blue')
plt.title('Support Vector Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()