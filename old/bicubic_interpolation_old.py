#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:17:18 2024

"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import numpy as np
import pandas as pd
import QuantLib as ql

def make_bicubic_ts(bicubic_vol, T, K):
    bicubic_ts = pd.DataFrame(
        [[bicubic_vol(t,k, True) for t in T] for k in K],
        columns=T,
        index=K)
    return bicubic_ts



def make_bicubic_functional(s,K,T,atm_volvec,derman_coefs):
    T = T.astype(int).tolist()
    K = K.astype(int).tolist()
    ql_T = ql.Array(T)
    ql_K = ql.Array(K)
    ql_vols = ql.Matrix(len(K),len(T),0.00)
    
    for i, k in enumerate(ql_K):
        for j, t in enumerate(ql_T):
            ql_vols[i][j] = atm_volvec[t] + derman_coefs[t]*(k-s)
    
    bicubic_vol = ql.BicubicSpline(ql_T, ql_K, ql_vols)
    return bicubic_vol




def make_vector_bicubic_functional(s,K,T,atm_volvec,derman_coefs):
    def make_bicubic_functional(s,K,T,atm_volvec,derman_coefs):
        T = T.astype(int).tolist()
        K = K.astype(int).tolist()
        ql_T = ql.Array(T)
        ql_K = ql.Array(K)
        ql_vols = ql.Matrix(len(K),len(T),0.00)
        
        for i, k in enumerate(ql_K):
            for j, t in enumerate(ql_T):
                ql_vols[i][j] = atm_volvec[t] + derman_coefs[t]*(k-s)
        
        bicubic_vol = ql.BicubicSpline(ql_T, ql_K, ql_vols)
        return bicubic_vol
    
    bicubic_vol = make_bicubic_functional(s,K,T,atm_volvec,derman_coefs)
    
    vbicubic_vol = np.vectorize(bicubic_vol)
    
    return vbicubic_vol












