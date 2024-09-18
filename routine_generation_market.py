# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:46:57 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import numpy as np
import pandas as pd
from itertools import product
from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
dividend_rate = settings[0]['dividend_rate']
risk_free_rate = settings[0]['risk_free_rate']

security_settings = settings[0]['security_settings']
s = security_settings[5]

ticker = security_settings[0]
lower_moneyness = security_settings[1]
upper_moneyness = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]

day_count = settings[0]['day_count']
calendar = settings[0]['calendar']
calculation_date = settings[0]['calculation_date']



from routine_collection import contract_details
calls = contract_details['calls']
puts = contract_details['puts']  

from routine_calibration_market import \
    put_heston_parameters, call_heston_parameters

from derman_test import derman_coefs, atm_volvec, derman_ts

# calls = calls[calls['days_to_maturity'].isin(atm_volvec.index)]
# puts = puts[puts['days_to_maturity'].isin(atm_volvec.index)]


def generate_features(K,T,s,flag):
    features = pd.DataFrame(
        product(
            [s],
            K,
            T,
            [flag]
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    return features

def map_features(df,parameters,flag):
    
    S = np.sort(np.array(df['spot_price'].unique()))
    byS = df.groupby('spot_price')
    option_features = pd.DataFrame()
    
    for s_idx, s in enumerate(S):
        
        dfs = byS.get_group(s)
        
        K = np.linspace(s*0.8,s*1.5,int(1e1))
        T = np.sort(dfs['days_to_maturity'].unique()).astype(int)
        
        initial_features = generate_features(K,T,s,flag)
        option_features = pd.concat([option_features,initial_features])
        
        feature_parameters = parameters.copy()
        feature_parameters['days_to_maturity'] = \
            feature_parameters['days_to_maturity'].astype(float)
        
        feature_params = feature_parameters.set_index(
            ['spot_price','days_to_maturity'])
        
        columns_to_map = [
            
            'v0', 'kappa', 'theta','rho','sigma',
            
            # 'volatility',
            
            # 'black_scholes',
            
            # 'heston'
            
            ]
        
        mapped_features = option_features.merge(
            feature_params[columns_to_map], 
            left_on=['spot_price', 'days_to_maturity'], 
            right_index=True, how='left')
        
        contracts_indexed = df.copy().set_index(
            ['spot_price','days_to_maturity'])
        
        mapped_features = mapped_features.merge(
            contracts_indexed[['risk_free_rate', 'dividend_rate']],
            left_on=['spot_price', 'days_to_maturity'], 
            right_index=True, how='left')
        
        def apply_derman_vols(row):
            try:
                s = row['spot_price']
                k = row['strike_price']
                t = row['days_to_maturity']
                moneyness = k-s
                b = derman_coefs.loc['b',t]
                atm_vol = atm_volvec[t]
                
                volatility = atm_vol + b*moneyness
                
                row['volatility'] = volatility
                return row
            except Exception:
                row['volatility'] = np.nan
                # print(f"no data {t} day maturity")
                return row
        
        mapped_features = mapped_features.apply(apply_derman_vols,axis=1)
        
        mapped_features = mapped_features.dropna()
        
    return mapped_features
    
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
# pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')

mapped_calls = map_features(calls, call_heston_parameters,"call")

from pricing import heston_price_vanilla_row, noisyfier
priced_features = mapped_calls.apply(heston_price_vanilla_row, axis = 1)

call_features = noisyfier(priced_features).dropna()

mapped_puts = map_features(puts, put_heston_parameters,"put")

priced_features = mapped_puts.apply(heston_price_vanilla_row, axis = 1)

put_features = noisyfier(priced_features).dropna()
put_features

features_dataset = pd.concat([mapped_calls,mapped_puts]).dropna(
    ).reset_index(drop=True)

print(f"\n{features_dataset}")
print(f"\n{features_dataset.describe}") 
