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
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')


data_files = dirdata()
term_structure_from_market = pd.DataFrame()
for file in data_files:
    try:
        df = pd.read_excel(file)
        df.columns = df.loc[1]
        df = df.iloc[2:,:].reset_index(drop=True)
        df = df.dropna()
        df = df.set_index('Strike')
        df_strikes = df.index.tolist()
        df_maturities = df['DyEx'].loc[df_strikes[0]].unique().tolist()
        calls = pd.concat([df.iloc[:, i:i+2] for i in range(
            0, df.shape[1], 4)], axis=1)
        callvols = calls['IVM']
        callvols.columns = df_maturities
        term_structure_from_market = pd.concat([term_structure_from_market,callvols])
        print(f"\n{file}:")
        print(f"{df_maturities}")
        print(f"center_strike: {np.median(df_strikes)}")
        print(f"count: {len(df_strikes)}")
    except Exception as e:
        print(f"\n{file}: {e}")
    except Exception:
        pass
    continue


strikes = np.sort(term_structure_from_market.index.unique())
maturities = np.sort(term_structure_from_market.columns.unique())
maturities = maturities[maturities>0]


# Create an empty DataFrame with strikes as the index and maturities as the columns
implied_vols_df = pd.DataFrame(index=strikes, columns=maturities)

# Loop through maturities and strikes
for i, maturity in enumerate(maturities):
    for j, strike in enumerate(strikes):
        try:
            value = term_structure_from_market.loc[strike, maturity]
            
            # Check if the value is numeric and not NaN
            if isinstance(value, (int, float)) and not np.isnan(value):
                implied_vols_df.loc[strike, maturity] = value
            else:
                print(f"\nInvalid value at Strike {strike}, Maturity {maturity}: {value}")
        except Exception as e:
            print(f"\nError at Strike {strike}, Maturity {maturity}: {e}")
        continue
ivdf = implied_vols_df.dropna(how='all')
ivdf = ivdf.dropna(how='all',axis=1)

maturities = ivdf.columns
maxmat = int(max(maturities))
minmat = int(min(maturities))
strikes =  ivdf.index
mink = int(min(strikes))
maxk = int(max(strikes))
S = int(np.median(strikes))
print(f"\n{ivdf}\n")
file_time = time.time()
file_datetime = datetime.fromtimestamp(file_time)
file_tag = file_datetime.strftime("%Y-%m-%d %H-%M-%S")
filename = f"SPX {file_tag} (S {S})(K {mink}-{maxk})(T {minmat}-{maxmat}).csv"
print(filename)





import QuantLib as ql
implied_vols_matrix = ql.Matrix(len(ivdf.index),len(ivdf.columns))

for i, maturity in enumerate(strikes):
    for j, strike in enumerate(maturities):
        implied_vols_matrix[i][j] = ivdf.iloc[i,j] 

print(f"\n{implied_vols_matrix}")

from settings import model_settings
ms = model_settings()
settings, ezprint = ms.import_model_settings()
dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']

expiration_dates = ms.compute_ql_maturity_dates(maturities)
S = np.median(ivdf.index)
black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, strikes,
    implied_vols_matrix, day_count)







