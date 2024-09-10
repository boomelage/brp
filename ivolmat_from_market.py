#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:25:46 2024

"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import QuantLib as ql

def extract_ivol_matrix_from_market(file):
    df = pd.read_excel(file)
    
    df.columns = df.loc[1]
    df = df.iloc[2:,:].reset_index(drop=True).dropna()
    df = df.set_index('Strike')
    strikes = df.index.tolist()
    maturities = df['DyEx'].loc[strikes[0]].unique().tolist()
    
    calls = pd.concat([df.iloc[:, i:i+2] for i in range(0, df.shape[1], 4)], axis=1)
    callvols = calls['IVM']
    callvols.columns = maturities
    
    
    implied_vols_matrix = ql.Matrix(len(strikes),len(maturities),float(0))
    
    for i, maturity in enumerate(maturities):
        for j, strike in enumerate(strikes):
            implied_vols_matrix[j][i] = callvols.iloc[j,i]
    
    return implied_vols_matrix, strikes, maturities, callvols

