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
        df = df.iloc[2:,:].reset_index(drop=True).dropna()
        df['Strike'] = df['Strike'].astype(int)
        df['DyEx'] = df['DyEx'].astype(int)
        df = df.set_index('Strike')
        df_strikes = df.index.tolist()
        df_maturities = df['DyEx'].loc[df_strikes[0]].unique().tolist()
        calls = pd.concat([df.iloc[:, i:i+2] for i in range(
            0, df.shape[1], 4)], axis=1)
        callvols = calls['IVM']
        callvols.columns = df_maturities
        term_structure_from_market = pd.concat([term_structure_from_market,callvols])
        print(f"\n{df_maturities}")
        print(f"center_strike: {np.median(df_strikes)}")
        print(f"count: {len(df_strikes)}")
    except Exception as e:
        print(f"\n{file}: {e}")
    continue
strikes = np.sort(term_structure_from_market.index.unique())
maturities = np.sort(term_structure_from_market.columns.unique())
maturities = maturities[maturities>0]
# term_structure_from_market.fillna(0,inplace=True)

term_structure_from_market

from settings import model_settings

mc = model_settings()

implied_vols_matrix = mc.extract_ivol_matrix_from_market(term_structure_from_market, maturities, strikes)