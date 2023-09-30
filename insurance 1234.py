import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
data=pd.read_csv('insurance.csv')
data.columns
data.info()
data.describe()

sns.heatmap(data.corr(),annot=True)
sns.scatterplot(x=data['age'],y=data['charges'])
sns.scatterplot(x=data['bmi'],y=data['charges'])

sns.boxplot(x=data['sex'],y=data['charges'])
sns.boxplot(x=data['children'],y=data['charges'])
sns.boxplot(x=data['smoker'],y=data['charges'])
sns.boxplot(x=data['region'],y=data['charges'])

columns=['sex','smoker','region']
encoder=LabelEncoder()
for column in columns:
    data[column]=encoder.fit_transform(data[column])
x=data.drop(['charges'],axis=1)
y=data['charges']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,
                                               random_state=0)
sel=SelectFromModel(Lasso(alpha=0.05))
sel.fit(x_train,y_train)
sel.get_support()
x.columns[sel.get_support()]

x_train=x_train.loc[:,['age', 'sex', 'bmi', 
                        'children', 'smoker', 'region']]
regressor1=Lasso(alpha=0.05)
regressor1.fit(x_train,y_train)

regressor1.coef_
regressor1.intercept_
y_pred=regressor1.predict(x_test)

np.sqrt(metrics.mean_squared_error(y_test, y_pred))
metrics.mean_absolute_error(y_test, y_pred)
metrics.r2_score(y_test,y_pred)