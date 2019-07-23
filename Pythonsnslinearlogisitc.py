# -*- coding: utf-8 -*-
"""
done Mykhailo Kravchenko
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

dataset=pd.read_csv("Social_Network_Ads.csv")

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

from sklearn.linear_model import LinearRegression

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

print(cm)

data=pd.read_csv("data.csv",header=0)
print(data.head(6))

data.info()

data.drop("Unnamed: 32",axis=1,inplace=True)
data.columns

data.drop("id",axis=1,inplace=True)
data["diagnosis"]=data["diagnosis"].map({'M':1,"B":0})

corr=data.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,cbar=True,square=True,cmap="coolwarm")
plt.show()
sns.countplot(data["diagnosis"],label="Count")
plt.show()

prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

train,test=train_test_split(data,test_size=0.3)
print(train.shape)
print(test.shape)


train_X=train[prediction_var]
train_Y=train.diagnosis
test_X=test[prediction_var]
test_y=test.diagnosis

logistic = LogisticRegression()
logistic.fit(train_X,train_Y)

from sklearn import metrics
temp=logistic.predict(test_X)
print(metrics.accuracy_score(temp,test_y))


