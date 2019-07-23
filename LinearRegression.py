# -*- coding: utf-8 -*-
"""
Created on Mykhail Kravchenko

"""
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("50-Startups.csv")
del dataset["State"]
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,:4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LabelEncoder = LabelEncoder()

X[:,2] = LabelEncoder.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[2]) 

X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split

X_train,x_test,y_train,y_test, = train_test_split(X, y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)


plt.scatter(y_pred, y_test)
plt.xlabel("y_pred")
plt.ylabel("y_test")

plt.show()



