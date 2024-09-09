# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 01:00:34 2024

creating the ivol matrix

"""

import QuantLib as ql
import numpy as np
def make_ivol_matrix(strikes,maturities,ivols,calculation_date):
    
    
    n_maturities = len(maturities)
    n_strikes = len(strikes)
    S = np.median(strikes)
    

    expiration_dates = np.empty(len(maturities),dtype=object)
    for i in range(len(expiration_dates)):
        expiration_dates[i] = calculation_date + \
            ql.Period(int(maturities[i]), ql.Days)


    implied_vol_matrix = ql.Matrix(n_strikes,n_maturities,float(0))
    for i in range(n_strikes):
        for j in range(n_maturities):
            implied_vol_matrix[i][j] = ivols.iloc[i,j]
          

    return n_maturities, n_strikes, S, expiration_dates, implied_vol_matrix
