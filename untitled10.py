# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:04:46 2023

@author: Misthi Bansal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import seaborn as sns
data=pd.read_csv(r"C:\Users\Misthi Bansal\Downloads\heart_disease_health_indicators_BRFSS2015.csv")
data.head(10)
data.head()
data.tail()
data.isna().sum()
data.describe()
data.columns
data.shape
#find corr and adapting vlaues
data=pd.read_csv(r"C:\Users\Misthi Bansal\Downloads\heart_disease_health_indicators_BRFSS2015.csv",na_values=[0])
data.isna().sum()
mean=data["HighBP"].mean()
data['HighBP']=data["HighBP"].replace(0,mean)

import matplotlib.pyplot as mlt
mlt.figure(figsize=(10,8))
sns.pairplot(data)
#ploting the graphs
mlt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)
columns=['HeartDiseaseorAttack', 'HighBP', 'HighChol', 'CholCheck', 'BMI',
       'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']
for i in columns:
    mlt.figure(figsize=(10,8))
    sns.boxplot(x=data['Income'],y=data[i])
    
mlt.figure(figsize=(10,8))    
mlt.scatter(x=data['HighBP'],y=data['Income'])    
mlt.scatter(x=data['BMI'],y=data['Income'])
#segregating input and output
x=data.drop(['Income'],axis=1)
y=data['Income']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20
                                               ,random_state=0)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
#####################
from sklearn import metrics
metrics.confusion_matrix(y_test,y_pred)
metrics.accuracy_score(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))
