#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:08:30 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
from data_query import dirdata
import pandas as pd
import numpy as np
import time
from datetime import datetime
# pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')


data_files = dirdata()
term_structure_from_market = pd.DataFrame()
for file in data_files:
    try:
        df = pd.read_excel(file)
        df.columns = df.loc[1]
        df = df.iloc[2:,:].reset_index(drop=True)
        
        
    #     df = df.dropna()
        callvols = df.set_index('Strike')
        df_strikes = df.index.tolist()
        df_maturities = df['DyEx'].loc[df_strikes[0]].unique().tolist()
        callvols = pd.concat([df.iloc[:, i:i+2] for i in range(
            0, df.shape[1], 4)], axis=1)
        # callvols = calls['IVM']
        # callvols.columns = df_maturities
        term_structure_from_market = pd.concat([term_structure_from_market,callvols])
    #     print(f"\n{file}:")
    #     print(f"{df_maturities}")
    #     print(f"center_strike: {np.median(df_strikes)}")
    #     print(f"count: {len(df_strikes)}")
    except Exception as e:
        print(f"\n{file}: {e}")
term_structure_from_market = term_structure_from_market.set_index('Strike')
strikes = np.sort(term_structure_from_market.index.unique())
maturities = np.array(term_structure_from_market['DyEx'].astype(float))
maturities = maturities[~np.isnan(maturities)].astype(int)
maturities = np.unique(maturities[maturities > 0])

term_structure_from_market = term_structure_from_market.reset_index()
term_structure_from_market



ts_columns = []
ivm_count = 1
dyex_count = 1

for col in term_structure_from_market.columns:
    if col == 'IVM':
        ts_columns.append(f'IVM_{ivm_count}')
        ivm_count += 1
    elif col == 'DyEx':
        ts_columns.append(f'DyEx_{dyex_count}')
        dyex_count += 1
    else:
        ts_columns.append(col)

term_structure_from_market.columns = ts_columns

df = term_structure_from_market

# Step 1: Reshape the DataFrame using pd.melt
df_melted = pd.melt(df, 
                    id_vars=['Strike'], 
                    value_vars=['IVM_1', 'IVM_2', 'IVM_3', 'IVM_4'],
                    var_name='IVM_label', 
                    value_name='IVM')

df_melted_dyex = pd.melt(df, 
                         id_vars=['Strike'], 
                         value_vars=['DyEx_1', 'DyEx_2', 'DyEx_3', 'DyEx_4'],
                         var_name='DyEx_label', 
                         value_name='DyEx')

# Combine the melted dataframes (assuming the same order)
df_combined = pd.concat([df_melted[['Strike', 'IVM']], df_melted_dyex['DyEx']], axis=1)

# Step 2: Drop rows where DyEx or IVM is NaN
df_combined = df_combined.dropna()

# Step 3: Set Strike and DyEx as the MultiIndex
df_indexed = df_combined.set_index(['Strike', 'DyEx'])

df_indexed = df_indexed.sort_index()


import QuantLib as ql

implied_vols_matrix = ql.Matrix(len(strikes),len(maturities),0)
for i, maturity in enumerate(maturities):
    for j, strike in enumerate(strikes):
        try:
            implied_vols_matrix[j][i] = float(df_indexed.xs((strike, maturity))['IVM'].iloc[0])
        except Exception:
            implied_vols_matrix[j][i] = 0
            
implied_vols_np = np.zeros((len(strikes), len(maturities)), dtype=float)
for i, maturity in enumerate(maturities):
    for j, strike in enumerate(strikes):
        implied_vols_np[j][i] = implied_vols_matrix[j][i]

implied_vols_df = pd.DataFrame(implied_vols_np)
implied_vols_df.index = strikes
implied_vols_df.columns = maturities

maxmat = int(max(maturities))
minmat = int(min(maturities))
strikes =  implied_vols_df.index
mink = int(min(strikes))
maxk = int(max(strikes))
S = int(np.median(strikes))
print(f"\n{implied_vols_df}\n")
file_time = time.time()
file_datetime = datetime.fromtimestamp(file_time)
file_tag = file_datetime.strftime("%Y-%m-%d %H-%M-%S")
filename = f"SPX {file_tag} (S {S})(K {mink}-{maxk})(T {minmat}-{maxmat}).csv"
implied_vols_df.to_csv(filename)




# from settings import model_settings
# ms = model_settings()
# settings, ezprint = ms.import_model_settings()
# dividend_rate = settings['dividend_rate']
# risk_free_rate = settings['risk_free_rate']
# calculation_date = settings['calculation_date']
# day_count = settings['day_count']
# calendar = settings['calendar']
# flat_ts = settings['flat_ts']
# dividend_ts = settings['dividend_ts']

# expiration_dates = ms.compute_ql_maturity_dates(maturities)
# S = np.median(strikes)
# black_var_surface = ql.BlackVarianceSurface(
#     calculation_date, calendar,
#     expiration_dates, strikes,
#     implied_vols_matrix, day_count)










