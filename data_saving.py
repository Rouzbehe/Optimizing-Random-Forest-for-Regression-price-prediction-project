"""
Created on Wed Sep 22 22:08:45 2018

@author: Rouzbeh Davoudi
"""

import csv
import pandas as pd
import pickle



def get_test_df():
    return pd.read_csv('C:/Users/rouzbeh/Desktop/New folder/test_features.csv')

def save_model(model):
    pickle.dump(model, open('C:/Users/rouzbeh/Desktop/New folder/RandomForest_classifier.pickle', 'wb'))

def load_model():
    return pickle.load(open('C:/Users/rouzbeh/Desktop/New folder/RandomForest_classifier.pickle', 'rb'))

def write_submission(predictions):
    writer = csv.writer(open('C:/Users/rouzbeh/Desktop/New folder/test_salaries.csv', 'w'), lineterminator="\n")
    test = get_test_df()
    rows = [x for x in zip(test["jobId"], predictions.flatten())]
    writer.writerow(("jobId", "Salary"))
    writer.writerows(rows)