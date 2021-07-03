# -*- coding: utf-8 -*-
"""



"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
ww2=pd.read_csv("C:/Users/ADMIN/Desktop/Siddhi/Summary of Weather.csv")
climate=pd.DataFrame(ww2)

climate.head()
climate.info()
climate.describe()
climate.columns

to_drop=['Precip','Date','STA','WindGustSpd','Snowfall','PoorWeather', 'PRCP', 'DR', 'YR','MO','DA','SPD', 'SNF', 'SND', 'FT', 'FB', 'FTI', 'ITH', 'PGT', 'TSHDSBRSGF', 'SD3', 'RHX', 'RHN', 'RVG', 'WTE']
climate.drop(to_drop,inplace=True,axis=1)

#checking missing values
climate.isna().sum()
#dropping na values
climate.dropna(axis=1)

climate.dtypes
#check what the below code does
climate=climate[~climate['MeanTemp'].isna()]
climate=climate[~climate['MaxTemp'].isna()]
climate=climate[~climate['MinTemp'].isna()]

climate.info()
climate.head()

#visualization
sns.countplot(x='MaxTemp', data=climate, palette='magma')
plt.title('Weather')

sns.regplot(x='MaxTemp', y='MinTemp', data=climate)
sns.heatmap(climate.corr())
plt.xticks(rotation=-45)

sns.pointplot(x='MaxTemp',y='MinTemp', data=climate)

sns.countplot(x='MaxTemp',data=climate)
sns.countplot(x='MinTemp',data=climate)
sns.distplot(climate['MinTemp'],kde = False)

sns.relplot(data=climate, x="MaxTemp", y="MinTemp",kind="line",ci=None)

sns.distplot(climate['MaxTemp'])

y=climate['MaxTemp']
X=climate['MinTemp']

#model making
X=climate.iloc[:,:1].values
y=climate.iloc[:, 2].values
print(X)
print(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size =0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#predicting
y_pred= regressor.predict(X_test)
print(y_pred)
#accuracy
from sklearn.metrics import r2_score
r2_score(y_pred,y_test)

#model accuracy = 0.9345227711348254
