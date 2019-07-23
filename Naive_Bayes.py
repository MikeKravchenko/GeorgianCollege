#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns


# In[8]:


# Data import
data = pd.read_csv('pacific.csv')
print(data.head(6))

# Data manipulation
data.Status = pd.Categorical(data.Status)
data['Status'] = data.Status.cat.codes
print(data.head())

# plotting Typhon class frequency
sns.countplot(data['Status'], label = 'Count')
plt.show()

# data wrangling
pred_columns = data[:]
pred_columns.drop(['Status'], axis = 1, inplace = True)
pred_columns.drop(['Event'], axis = 1, inplace = True)
pred_columns.drop(['Latitude'], axis = 1, inplace = True)

pred_columns.drop(['Longitude'], axis = 1, inplace = True)
pred_columns.drop(['ID'], axis = 1, inplace = True)
pred_columns.drop(['Name'], axis = 1, inplace = True)
pred_columns.drop(['Date'], axis = 1, inplace = True)
pred_columns.drop(['Time'], axis = 1, inplace = True)
prediction_var = pred_columns.columns
print(list(prediction_var))

# Train - Test Split
train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)
# Creating Response and Target Variable
train_X = train[prediction_var]
train_y = train['Status']
print(list(train.columns))

test_X = test[prediction_var]
test_y = test['Status']
# Confusion Matrix
modelGNB = GaussianNB()
modelGNB.fit(train_X, train_y)
y_pred_gnb = modelGNB.predict(test_X)
cnf_matrix_gnb  = confusion_matrix(test_y, y_pred_gnb)
print(cnf_matrix_gnb)
print("Number of mislabled points out of a total %d points: %d"%(data.shape[0], (test_y != y_pred_gnb).sum()))

#

#start from slide 25
from sklearn import svm
model = svm.SVC(kernel ='linear', C= 1, gamma = 1)

###
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.use("Agg")

from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn import svm
import random

data = pd.read_csv('pacific.csv')
print(data.head(6))

data.Status = pd.Categorical(data.Status)
data['Status'] = data.Status.cat.codes
print(data.head())

import seaborn as sns
sns.countplot(data['Status'], label ="Count" )
plt.show()

random.seed(2)
pred_columns = data[:]
pred_columns.drop(['Status'], axis = 1, inplace = True)
pred_columns.drop(['Event'], axis = 1, inplace = True)
pred_columns.drop(['Latitude'], axis = 1, inplace = True)
pred_columns.drop(['Longitude'], axis = 1, inplace = True)
pred_columns.drop(['ID'], axis = 1, inplace = True)
pred_columns.drop(['Name'], axis = 1, inplace = True)
pred_columns.drop(['Date'], axis = 1, inplace = True)
pred_columns.drop(['Time'], axis = 1, inplace = True)
prediction_var = pred_columns.columns
print(list(prediction_var))

train, test = train_test_split(data, test_size = 0.3)
print(train.shape)
print(test.shape)

train_X = train[prediction_var]
train_y = train['Status']
print(list(train.columns))

test_X = test[prediction_var]
test_y = test['Status']

model = svm.SVC(kernel = 'linear')
# ### ERROR BELOW
model.fit(train_X, train_y)
predicted = model.predict(test_X)

print("SVM accuray:", accuracy_score(test_y, predicted))












