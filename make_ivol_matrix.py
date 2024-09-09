# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 01:00:34 2024

creating the ivol matrix

"""

import QuantLib as ql
import numpy as np

calculation_date = ql.Date.todaysDate()

def make_ivol_matrix(
        strikes,maturities,ivol_table,calculation_date,n_strikes,n_maturities): 
    
    
    expiration_dates = np.empty(len(maturities),dtype=object)
    for i in range(len(expiration_dates)):
        expiration_dates[i] = calculation_date + \
            ql.Period(int(maturities[i]), ql.Days)


    implied_vol_matrix = ql.Matrix(n_strikes,n_maturities,float(0))
    for i in range(n_maturities):
        for j in range(n_strikes):
            implied_vol_matrix[j][i] = ivol_table[i][j]
          

    return expiration_dates, implied_vol_matrix