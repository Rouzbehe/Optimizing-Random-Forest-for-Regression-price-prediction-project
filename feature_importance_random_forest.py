# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:58:44 2018

@author: Rouzbeh Davoudi
"""
import numpy as np
from math import sqrt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/rouzbeh/Desktop/New folder/train_features.csv')
train_salaries = pd.read_csv('C:/Users/rouzbeh/Desktop/New folder/train_salaries.csv')

#train=train.sample(frac=0.01,random_state=1)
#train_salaries=train_salaries.sample(frac=0.01,random_state=1)

#train[pd.isnull(train)]  = 'NaN'
msk = np.random.rand(len(train)) < 0.8

features=list(range(1,8))

X_train=train[msk].iloc[:,features]
y_train=train_salaries[msk].iloc[:,1]

X_valid=train[~msk].iloc[:,features]
y_valid=train_salaries[~msk].iloc[:,1]



X_train_ordinal = X_train.values
X_valid_ordinal = X_valid.values
les = []
l = LogisticRegression()
r = RandomForestRegressor(n_estimators=288, max_depth=64, min_samples_split=10, min_samples_leaf=4, max_features='auto', bootstrap=True, n_jobs=-1)

for i in range(X_train_ordinal[:,0:6].shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(train.iloc[:,features].iloc[:, i])
    les.append(le)
    X_train_ordinal[:, i] = le.transform(X_train_ordinal[:, i])
    X_valid_ordinal[:, i] = le.transform(X_valid_ordinal[:, i])   
         
r.fit(X_train_ordinal,y_train)
#y_pred = l.predict_proba(X_valid_ordinal)


feat_importances = pd.DataFrame(r.feature_importances_,
                                    index = X_train.columns,
                                     columns=['importance']).sort_values('importance',ascending=False)


feature_importances = pd.Series(r.feature_importances_, index=X_train.columns)
feature_importances.nlargest(8).plot(kind='barh')
plt.show()




y_pred = r.predict(X_valid_ordinal)

fig, ax = plt.subplots()
ax.scatter(y_valid, y_pred)
ax.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=4)
ax.set_xlabel('True Salaries')
ax.set_ylabel('Predicted Salaries')
plt.show()  
    
print(sqrt(mean_squared_error(y_valid, y_pred)))  #19.68
print(r2_score(y_valid, y_pred))  # 0.742

