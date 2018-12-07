"""
Created on Wed Sep 22 22:08:45 2018

@author: Rouzbeh Davoudi
"""


import data_saving
import pandas as pd

test = pd.read_csv('C:/Users/rouzbeh/Desktop/New folder/test_features.csv')
features=list(range(1,8))
X_test=test.iloc[:,features]

def main():
    print("Loading the classifier")
    classifier = data_saving.load_model()
    
    print("Making predictions") 

    predictions = classifier.predict(X_test)   
    predictions = predictions.reshape(len(predictions), 1)

    print("Writing predictions to file")
    data_saving.write_submission(predictions)

if __name__=="__main__":
    main()