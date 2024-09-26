# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
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

df = training_data

def compute_moneyness_row(df):
    df['moneyness'] = 0.00
    for i, row in df.iterrows():
        if row['w'] == 'call':
            df.at[i,'moneyness'] = df.at[i,'spot_price']/df.at[i,'strike_price'] - 1
        elif row['w'] == 'put':
            df.at[i,'moneyness'] = df.at[i,'strike_price']/df.at[i,'spot_price'] - 1
        else:
            raise ValueError('put/call flag error')
    return df
    
print(training_data)

training_data.columns