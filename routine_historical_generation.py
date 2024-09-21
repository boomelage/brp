# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:10:31 2024

@author: boomelage
"""

"""
generation routine
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import pandas as pd
from itertools import product

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

historical_option_data = pd.DataFrame()
# for i, row in historical_impvols.iterrows():
    
    
row = historical_impvols.iloc[0]
s = row['spot_price']
dtdate = row['date']

calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)

expiry_dates = np.array([
      calculation_date + ql.Period(30,ql.Days), 
      calculation_date + ql.Period(60,ql.Days), 
      calculation_date + ql.Period(3,ql.Months), 
      # calculation_date + ql.Period(6,ql.Months),
      # calculation_date + ql.Period(12,ql.Months), 
      # calculation_date + ql.Period(18,ql.Months), 
      # calculation_date + ql.Period(24,ql.Months)
      ],dtype=object)
T = expiry_dates - calculation_date

call_K = np.linspace(s, s*1.005, 5)
put_K  = np.linspace(s, s*0.995, 5)

calls = generate_features(call_K, T, s)
calls = calls[calls['days_to_maturity'].isin(T)].copy()
calls['w'] = 'call'
calls['moneyness'] = calls['spot_price'] - calls['strike_price']

puts = generate_features(put_K, T, s)
puts = puts[puts['days_to_maturity'].isin(T)].copy()
puts['w'] = 'put'
puts['moneyness'] = puts['strike_price'] - puts['spot_price']

calls = calls.apply(bicubic_vol_row,axis=1)
puts = puts.apply(bicubic_vol_row,axis=1)

features = pd.concat([calls,puts],ignore_index=True)
contract_details = features.copy()

contract_details = contract_details.apply(bicubic_vol_row,axis=1)

contract_details


contract_details['risk_free_rate'] = 0.04
contract_details['dividend_rate'] = row['dividend_rate']

heston_parameters = calibrate_heston(contract_details,s)


pricing_spread = 0.005
call_K_interp = np.linspace(s, s*(1+pricing_spread),100)
put_K_interp = np.linspace(s*(1-pricing_spread),s,100)


train_calls = generate_train_features(call_K,T,s,['call'])
