# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from data_query import dirdatacsv
from settings import model_settings
ms = model_settings()
csvs = dirdatacsv()

training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)
    
training_data = training_data.drop(
    columns=training_data.columns[0]).drop_duplicates()


def compute_moneyness(df):
    df.loc[
        df['w'] == 'call', 
        'moneyness'
        ] = df['spot_price'] / df['strike_price'] - 1
    df.loc[
        df['w'] == 'put', 
        'moneyness'
        ] = df['strike_price'] / df['spot_price'] - 1
    return df

training_data = compute_moneyness(training_data)
training_data = training_data[abs(training_data['moneyness'])>=0.03].reset_index(drop=True)

pd.set_option("display.max_columns",None)
print(f"\n{training_data.describe()}\n")
pd.reset_option("display.max_columns")
