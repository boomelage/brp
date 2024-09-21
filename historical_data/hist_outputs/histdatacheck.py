#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:36:20 2024

@author: doomd
"""

import pandas as pd

df = pd.read_csv(r"/Users/doomd/git/brp/historical_data/2024-09-21 17-35-10.csv")
df = df.drop(columns = df.columns[0])

df = df.iloc[:100,:]

for i, row in df.iterrows():
    if row['w'] == 'call':
        df.loc[i, 'moneyness'] = row['spot_price'] - row['strike_price']
    elif row['w'] == 'put':
        df.loc[i, 'moneyness'] = row['strike_price'] - row['spot_price']
    else:
        print('flag error')
        
df.describe()