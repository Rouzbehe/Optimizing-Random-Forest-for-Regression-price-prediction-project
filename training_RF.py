# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:08:45 2018

@author: Rouzbeh Davoudi

This code uses optimized random forest found by training.py file, and save the random forest model.
"""
from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from DataFrameImputer import DataFrameImputer
from future_encoders_short import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import data_saving
   
train = pd.read_csv('C:/Users/rouzbeh/Desktop/New folder/train_features.csv')
train_salaries = pd.read_csv('C:/Users/rouzbeh/Desktop/New folder/train_salaries.csv')

#train=train.sample(frac=0.01,random_state=1)
#train_salaries=train_salaries.sample(frac=0.01,random_state=1)

#train.replace('NONE', np.nan, inplace=True)

msk = np.random.rand(len(train)) < 0.8

features=list(range(1,8))

X_train=train[msk].iloc[:,features]
y_train=train_salaries[msk].iloc[:,1]

X_valid=train[~msk].iloc[:,features]
y_valid=train_salaries[~msk].iloc[:,1]

# all of training data
X_train_all=train.iloc[:,features]
y_train_all=train_salaries.iloc[:,1]


def data_prepartion():
    
    num_attribs = ["yearsExperience", "milesFromMetropolis"]
    cat_attribs = ["companyId", "jobType", "degree", "major", "industry"]
    num_pipeline = Pipeline([
            ('imputer', DataFrameImputer(num_attribs)),
            ('std_scaler', StandardScaler()),
        ])
    
    cat_pipeline = Pipeline([
            ('imputer', DataFrameImputer(cat_attribs)),
            ('label_binarizer', OneHotEncoder()),
        ])
    
    data_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline)
        ])
    return data_pipeline




def randomforest_opt():
    data_pipeline=data_prepartion()
    X_train_prepared=data_pipeline.fit_transform(X_train)
    X_valid_prepared=data_pipeline.fit_transform(X_valid)

    # the hyperarameters used here were found by "training.py" file.
    rf_random=RandomForestRegressor(n_estimators=288, max_depth=64, min_samples_split=10, min_samples_leaf=4, max_features='auto', bootstrap=True, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train_prepared, y_train)
    

    
    y_pred=rf_random.predict(X_valid_prepared)
    
    print(sqrt(mean_squared_error(y_valid, y_pred)))  
    print(r2_score(y_valid, y_pred))  
    
    
    fig, ax = plt.subplots()
    ax.scatter(y_valid, y_pred)
    ax.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'k--', lw=4)
    ax.set_xlabel('True Salaries')
    ax.set_ylabel('Predicted Salaries')
    plt.show()    
    
    return rf_random    

def get_pipeline():
    data_pipeline=data_prepartion()
    best_randomforest=randomforest_opt()
    steps = [("extract_features", data_pipeline),
             ("classify with Optimized Random Forest", best_randomforest)]
    return Pipeline(steps)


def main():
    print("Preparing Data, trasnfroming Features and training model")
    classifier = get_pipeline()
    classifier.fit(X_train_all, y_train_all)

    print("Saving the classifier")
    data_saving.save_model(classifier)
    
if __name__=="__main__":
    main()

