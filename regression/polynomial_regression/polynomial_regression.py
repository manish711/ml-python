# -*- coding: utf-8 -*-
"""
@author: manishnarang
Polynomial Regression
"""
#Importing Libraries
import numpy as np #contains mathematical tools.
import matplotlib.pyplot as plt #to help plot nice charts.
import pandas as pd #to import data sets and manage data sets.

#Importing Data Set - difference between the independent variables and the dependent variables.
dataSet = pd.read_csv('Position_Salaries.csv')
X = dataSet.iloc[:, 1:-1].values
Y = dataSet.iloc[:, -1].values

#Split the training set and test set
#No because we have very few observations we actually won't split the data set into a training set and a test set because what we want to get directly is actually that prediction of the position leve. In this case, it is between 6 and 7. So we actually need to get the maximum data we can.

#Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polyFeature = PolynomialFeatures(degree = 4) #for higher degree;smoother the curve
X_poly = polyFeature.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

#Visualising the Linear Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg2.predict(polyFeature.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict([[6.5]]) #What does this double pair of square brackets mean. Well the first pair of square brackets here corresponds to the first dimension and the second pair of square brackets here correspond to the second dimension. So the first dimension is actually corresponding to the rows in your array. And the second dimension is corresponding to your column.

#Predicting a new result with Linear Regression
lin_reg2.predict(polyFeature.fit_transform([[6.5]]))