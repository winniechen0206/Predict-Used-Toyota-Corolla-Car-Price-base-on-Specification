# -*- coding: utf-8 -*-
"""
Created on Sun Mar 1 18:10:16 2020

@author: wchen
"""
# In[1]:
# Importing the libraries
import pandas as pd
import os

# Importing the dataset
os.chdir("C:/Users/wchen/Downloads")
dataset = pd.read_csv('ToyotaCorolla.csv')

dataset = dataset[['Age_08_04', 'KM', 'Fuel_Type', 'HP', 'Automatic', 'Doors', 'Quarterly_Tax',
'Mfr_Guarantee', 'Guarantee_Period', 'Airco', 'Automatic_airco', 'CD_Player',
'Powered_Windows', 'Sport_Model', 'Tow_Bar', 'Price']]
dataset.head()

# In[2]:
# Encoding categorical data
dataset = pd.get_dummies(dataset, columns=['Fuel_Type'], prefix = ['Fuel'])

# In[3]:
# Prepare Features and Labels
X = dataset.loc[:,dataset.columns != 'Price']
y = dataset.loc[:,dataset.columns == 'Price']

# In[4]:
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# In[5]:
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Label Scaling
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


# In[6]:
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

# In[7]:
# Part 3 - Making predictions and evaluating the model
from math import sqrt

# Predicting the Test set results
y_pred = classifier.predict(X_test)
rmse = sqrt(sum((y_test-y_pred)**2)/len(y_test))
print(rmse) 
"""
Out[63]: 0.06347389491319165
"""

# Predicting the Training set results
y_pred = classifier.predict(X_train)
rmse = sqrt(sum((y_train-y_pred)**2)/len(y_train))
print(rmse)
"""
Out[64]: 0.06188155141872638
"""

# In[8]:
# Modify the model to single layer with 5 nodes

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

# In[9]:
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
rmse = sqrt(sum((y_test-y_pred)**2)/len(y_test))
print(rmse) 
"""
Out[63]: 0.05157386947653265
"""

# Predicting the Training set results
y_pred = classifier.predict(X_train)
rmse = sqrt(sum((y_train-y_pred)**2)/len(y_train))
print(rmse)
"""
Out[64]: 0.04982500227298357
"""

# In[10]:
# Modify the model to two layers with 5 nodes in each layer

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

# In[9]:
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
rmse = sqrt(sum((y_test-y_pred)**2)/len(y_test))
print(rmse) 
"""
Out[63]: 0.04632696197448908
"""

# Predicting the Training set results
y_pred = classifier.predict(X_train)
rmse = sqrt(sum((y_train-y_pred)**2)/len(y_train))
print(rmse)
"""
Out[64]: 0.04661587637538337
"""