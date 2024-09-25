# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:30:49 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
from data_query import dirdatacsv
csvs = dirdatacsv()
import pandas as pd
from mlop import mlop


mlop=mlop(user_dataset=training_data)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in mlop.categorical_features:
    feature_vector = training_data[feature]
    le.fit(feature_vector)
    training_data[feature] = le.transform(feature_vector)
    
