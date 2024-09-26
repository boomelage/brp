# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pandas as pd
from data_query import dirdatacsv
from settings import model_settings, compute_moneyness
os.chdir(current_dir)

ms = model_settings()
csvs = dirdatacsv()

training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)
    
training_data = training_data.drop(
    columns=training_data.columns[0]).drop_duplicates()

training_data = compute_moneyness(training_data)

# training_data = training_data[
#     (abs(training_data['moneyness'])>=0.02)&
#     (abs(training_data['moneyness'])<=0.1)
#     ].reset_index(drop=True)

# training_data = training_data[
#     (abs(training_data['days_to_maturity'])>=0)&
#     (abs(training_data['days_to_maturity'])<=3)
#     ].reset_index(drop=True)

pd.set_option("display.max_columns",None)
print(f"\n{training_data.describe()}\n")
print(f"\ntypes:\n{training_data['w'].unique()}\n")
print(f"\nmaturities:\n{training_data['days_to_maturity'].unique()}\n")
print(f"\nstrikes:\n{training_data['strike_price'].unique()}\n")
print(f"\nspot(s):\n{training_data['spot_price'].unique()}\n")
pd.reset_option("display.max_columns")
