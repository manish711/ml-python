# -*- coding: utf-8 -*-
"""
@author: manishnarang
"""
#Importing Libraries
import numpy as np #contains mathematical tools.
import matplotlib.pyplot as plt #to help plot nice charts.
import pandas as pd #to import data sets and manage data sets.

#Importing Data Set - difference between the independent variables and the dependent variables.
dataSet = pd.read_csv('Data.csv')
X = dataSet.iloc[:, :-1].values
Y = dataSet.iloc[:, 3].values
print(X)
print(Y)

#Missing Data Values - the most common idea to handle missing data is to take the mean of the columns.  
from sklearn.impute import SimpleImputer
missingValues = SimpleImputer(missing_values=np.nan, strategy="mean")
missingValues = missingValues.fit(X[:, 1:3])
X[:, 1:3] = missingValues.transform(X[:, 1:3])
print(X)

#Categorical Data - Since machine learning models are based on mathematical equations, it would cause some problem if we keep the text in columns and the categorical variables in the equations because we would only want numbers in the equations.
#Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
Y = labelEncoder_X.fit_transform(Y)
print(Y)

#Feature Scaling - So that we don't get this sort of problem with a huge number here dominating a smaller number here so that eventually the smaller number doesn't exist.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

#Split the training set and test set - a test set on which we test the performance of this machine learning model and the performance on the test set shouldn't be that different from the performance on the training sets because this would mean that the machine learning models understood well the correlations and didn't learn them by heart.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

"""
P.S: No need for print statements as IDE (like spyder) provide Variable Explorer
"""

