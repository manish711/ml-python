# -*- coding: utf-8 -*-
"""
@author: manishnarang
Decision Tree Regression
"""
#Importing Libraries
import numpy as np #contains mathematical tools.
import matplotlib.pyplot as plt #to help plot nice charts.
import pandas as pd #to import data sets and manage data sets.

#Importing Data Set - difference between the independent variables and the dependent variables.
dataSet = pd.read_csv('Position_Salaries.csv')
X = dataSet.iloc[:, 1:-1].values
Y = dataSet.iloc[:, -1].values

#Train the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
decisionTreeRegressor = DecisionTreeRegressor(random_state = 0)
decisionTreeRegressor.fit(X, Y)

#Predicting a new result
decisionTreeRegressor.predict([[6.5]])

#Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, decisionTreeRegressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()