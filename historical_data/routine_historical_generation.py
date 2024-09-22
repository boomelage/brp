# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:10:31 2024

generation routine
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join('term_structure',parent_dir))
import pandas as pd
import numpy as np
import QuantLib as ql
import time
from datetime import datetime
from itertools import product
from bicubic_interpolation import bicubic_vol_row, make_bicubic_functional
from routine_calibration_global import calibrate_heston
from pricing import noisyfier
from settings import model_settings
from derman_test import call_dermans, make_derman_surface
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import collect_historical_data


# pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

# pd.reset_option("display.max_rows")
# pd.reset_option("display.max_columns")

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T,
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
                  ])
    return features

def generate_train_features(K,T,s,flag):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            flag
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    return features


"""
historical generation routine
"""
historical_data = collect_historical_data()

historical_option_data = pd.DataFrame()
for i, row in historical_data.iterrows():
    try:
        s = row['spot_price']
        dtdate = row['date']
        print(f"\n\n{dtdate}")
        calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)
        expiry_dates = np.array([
                calculation_date + ql.Period(30,ql.Days), 
                calculation_date + ql.Period(60,ql.Days), 
                calculation_date + ql.Period(3,ql.Months), 
                calculation_date + ql.Period(6,ql.Months),
                calculation_date + ql.Period(12,ql.Months), 
                # calculation_date + ql.Period(18,ql.Months), 
                # calculation_date + ql.Period(24,ql.Months)
              ],dtype=object)
        
        T = expiry_dates - calculation_date
        
        
        
        """
        calibration dataset construction
        """
        n = 4
        call_K = np.linspace(s*1.01,s*1.025, n)
        put_K  = np.linspace(s*0.975, s*0.99, n)
        
        calls = generate_features(call_K, T, s)
        calls = calls[calls['days_to_maturity'].isin(T)].copy()
        calls['w'] = 'call'
        calls['moneyness'] = calls['spot_price'] - calls['strike_price']
        
        puts = generate_features(put_K, T, s)
        puts = puts[puts['days_to_maturity'].isin(T)].copy()
        puts['w'] = 'put'
        puts['moneyness'] = puts['strike_price'] - puts['spot_price']

        derman_T = [30, 60, 95, 186, 368]
        atm_volvec = row[row.index[1:-4]]
        atm_volvec.index = derman_T
        
        derman_K = np.unique(np.array([put_K, call_K],dtype=float).flatten())
        derman_ts = make_derman_surface(atm_volvec, call_dermans, derman_K, s)
        
        bicvol = make_bicubic_functional(derman_ts,list(derman_K),list(derman_T))
        
        
        features = pd.concat([calls,puts],ignore_index=True)
        features['dividend_rate'] = row['dividend_rate']
        
        def apply_bicvol(row):
            t = row['days_to_maturity']
            k = row['strike_price']
            volatility = bicvol(t,k, allowExtrapolation = False)
            row['volatility'] = volatility
            return row
        
        
        features = features.apply(apply_bicvol, axis = 1)
        
        
        """
        calibration output (assuming fixed risk free rate for now)
        """
        
        features['risk_free_rate'] = 0.04
        heston_parameters = calibrate_heston(features, s, calculation_date)
        print('calibrated')
        
        """
        generation
        """
        train_T = np.arange(1,T[1],1)
        n = 15
        
        # call_K  = np.linspace(s*1.005, s*1.05, n)
        # call_features = generate_train_features(call_K,train_T,s,['call'])
        
        put_K  = np.linspace(s*0.99, s*0.991, n)
        put_features = generate_train_features(put_K,train_T,s,['put'])
        features = put_features
        
        # features = pd.concat(
        #     [call_features,put_features],ignore_index=True).reset_index(drop=True)   
        
        features['sigma'] = heston_parameters['sigma'].iloc[0]
        features['theta'] = heston_parameters['theta'].iloc[0]
        features['kappa'] = heston_parameters['kappa'].iloc[0]
        features['rho'] = heston_parameters['rho'].iloc[0]
        features['v0'] = heston_parameters['v0'].iloc[0]
        features['avgAbsRelErr'] = heston_parameters['avgAbsRelErr'].iloc[0]
        features['risk_free_rate'] = 0.04
        features['dividend_rate'] = row['dividend_rate']
        features['days_to_maturity'] = features['days_to_maturity'].astype(int)

        heston_features = features.apply(ms.heston_price_vanilla_row,axis=1)
        ml_data = noisyfier(heston_features)
        historical_option_data = pd.concat([historical_option_data,ml_data])
        print(f"\n{historical_option_data.describe()}")
        print(f"\n{i}/{historical_data.shape[0]+1}")
        
    except Exception as e:
        print(f"\n\n\n\nerror: {calculation_date}\ndetails: {e}\n\n\n\n")
        pass
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

print(f"\n{historical_option_data.describe()}")
file_time = time.time()
file_dt = datetime.fromtimestamp(file_time)
file_timetag = file_dt.strftime("%Y-%m-%d %H-%M-%S")
historical_option_data.to_csv(f"hist_outputs/{file_timetag}.csv")
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
