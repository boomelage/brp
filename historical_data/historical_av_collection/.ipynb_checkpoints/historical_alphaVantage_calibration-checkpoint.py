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


def clean_term_structure(ivol_df,w):
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
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)
from routine_calibration_global import calibrate_heston
os.chdir(current_dir)

store = pd.HDFStore([file for file in os.listdir() if file.endswith('.h5')][0])

dates = []
for key in store.keys():
    dates.append(key[key.find('date')+5:])

dates = pd.Series(dates).drop_duplicates()

datetimedates = pd.to_datetime(dates,format='%Y_%m_%d')
start_date = min(datetimedates).strftime('%Y_%m_%d')
end_date = max(datetimedates).strftime('%Y_%m_%d')

w = 'call'
r = 0.04
g = 0.02

av_features = pd.DataFrame()
errors = pd.Series()
for date in dates:
    try:
        calculation_datetime = datetime.strptime(date, "%Y_%m_%d")
        calculation_date = ql.Date(
            calculation_datetime.day,
            calculation_datetime.month,
            calculation_datetime.year
            )
        
        ts = store[f"{w}/surface/date_{date}"]
        contracts = store[f"{w}/contracts/date_{date}"]
        s = float(contracts['spot_price'].unique()[0])
        K = ts.index.tolist()
        T = ts.columns.tolist()
        
        surface = clean_term_structure(ts,w)  
        
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
        
        K = np.linspace(
            min(K),
            max(K),
            5)
        maxT = 180
        minT = 30
        T = np.arange(
            minT,
            maxT+1,
            int((maxT-minT)/5)
        )
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
                        'relative_error',
                        ]
                    )
                
                ], ignore_index = True
            )
    except Exception as e:
        print(f"bad data for {key}\nerror: {e}")
        pass
store.close()
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

av_features.to_csv(os.path.join(
    current_dir,
    'historical_av_calibrations',
    f'av_calibrated {start_date}_{end_date}.csv'
    )
)



