# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:08:45 2018

@author: Rouzbeh Davoudi
"""

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin


#    
class DataFrameImputer(TransformerMixin):

   def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
   def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

   def transform(self, X, y=None):
        return X[self.attribute_names].fillna(self.fill).values
    