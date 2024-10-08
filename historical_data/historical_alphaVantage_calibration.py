#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:57:16 2024

@author: doomd
"""
import os
import sys
import pandas as pd
import numpy as np
import QuantLib as ql
from scipy import interpolate
from itertools import product
from datetime import datetime


def clean_term_structure(chain,date_key,w):
    ivol_df = chain[date_key][w]['surface']
    ivol_df = ivol_df.dropna(how='all',axis=0).dropna(how='all',axis=1)
    strikes = ivol_df.iloc[:,0].dropna().index
    ivol_df = ivol_df.loc[strikes,:].copy()
    T = ivol_df.columns.tolist()
    K = ivol_df.index.tolist()
    ivol_array = ivol_df.to_numpy()
    x = np.arange(0, ivol_array.shape[1])
    y = np.arange(0, ivol_array.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(ivol_array)
    xx, yy = np.meshgrid(x, y)
    
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    
    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                method='cubic')
    vol_surf = pd.DataFrame(
        GD1,
        index = K,
        columns = T
    ).copy()
    
    vol_surf = ivol_df.loc[:,ivol_df.columns>0].dropna(how='any', axis=0).copy()
    return vol_surf



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.chdir(current_dir)

from routine_calibration_global import calibrate_heston
from historical_alphaVantage_collection import chain

w = 'puts'
r = 0.04
g = 0.02

date_keys = []

for key in chain.keys():
    date_keys.append(key)

# key = date_keys[0]

"""
loop start
"""

av_features = pd.DataFrame()
errors = pd.Series()
for key in chain.keys():
    calculation_datetime = datetime.strptime(key, "%Y-%m-%d")
    surface = clean_term_structure(chain,key,w)

    """
    loop prototype
    """

    link = chain[key]
    calculation_datetime = datetime.strptime(key, "%Y-%m-%d")
    calculation_date = ql.Date(
        calculation_datetime.day,
        calculation_datetime.month,
        calculation_datetime.year
        )
    surface = clean_term_structure(chain,key,w)  
    
    
    T = surface.columns.tolist()
    K = surface.index.tolist()
    
    
    vol_matrix = ql.Matrix(
        len(K),
        len(T),
        0.0)
    for i,k in enumerate(surface.index.tolist()):
        for j,t in enumerate(surface.columns.tolist()):
            vol_matrix[i][j] = float(surface.loc[k,t])
    bicubic_vol = ql.BicubicSpline(
        surface.columns.tolist(),
        surface.index.tolist(),
        vol_matrix
        )
    
    s = float(link[w]['contracts']['spot_price'].unique()[0])
    K = np.linspace(
        min(K),
        max(K),
        5)
    T = np.arange(14,60,7)
    
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            [calculation_datetime]
            ),
        columns = [
            'spot_price','strike_price','days_to_maturity',
            'calculation_date'
            ]
        )
    
    def apply_vol(t,k):
        vol = bicubic_vol(float(t),float(k),False)
        return vol
    
    apply_vols = np.vectorize(apply_vol)
    features['volatility'] = apply_vols(
        features['days_to_maturity'],
        features['strike_price'],
        )

    heston_parameters = calibrate_heston(
        features, s, r, g, calculation_date)
    
    if heston_parameters['relative_error']<0.2:
        av_features = pd.concat(
            [
                av_features, 
                
                pd.DataFrame(
                    np.array([[
                        s,
                        calculation_datetime,
                        heston_parameters['theta'],
                        heston_parameters['rho'],
                        heston_parameters['kappa'],
                        heston_parameters['eta'],
                        heston_parameters['v0'],
                        heston_parameters['relative_error'],
                        ]],dtype = object),
                    
                    columns = [
                        'spot_price',
                        'calculation_date',
                        'theta',
                        'rho',
                        'kappa',
                        'eta',
                        'v0',
                        'relative_error'
                        ]
                    )
                
                ], ignore_index = True
            )
    else:
        pass
av_features = av_features.reset_index(drop=True)
av_features['spot_price'] = pd.to_numeric(
    av_features['spot_price'], errors = 'coerce')
av_features['theta'] = pd.to_numeric(
    av_features['theta'], errors = 'coerce')
av_features['rho'] = pd.to_numeric(
    av_features['rho'], errors = 'coerce')
av_features['kappa'] = pd.to_numeric(
    av_features['kappa'], errors = 'coerce')
av_features['eta'] = pd.to_numeric(
    av_features['eta'], errors = 'coerce')
av_features['v0'] = pd.to_numeric(
    av_features['v0'], errors = 'coerce')
av_features['relative_error'] = pd.to_numeric(
    av_features['relative_error'], errors = 'coerce')
pd.set_option('display.max_columns',None)
print(av_features.describe())



