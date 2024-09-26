# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
import time
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from data_query import dirdatacsv

csvs = dirdatacsv()

training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)
    
training_data = training_data.drop(
    columns=training_data.columns[0]).drop_duplicates().reset_index(drop=True)
print(training_data)