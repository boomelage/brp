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
import pandas as pd
from itertools import product
from pricing import noisyfier
from settings import model_settings
from routine_calibration_global import calibrate_heston
from bicubic_interpolation import bicubic_vol_row
ms = model_settings()
import numpy as np


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



s = ms.s

call_K = ms.calibration_call_K
put_K = ms.calibration_put_K

n_strikes = int(1e4)

max_maturity = 31
T = np.arange(min(ms.T),max_maturity,1).astype(int)

n_contracts = int(len(T)*n_strikes*2)

print(f"\n\npricing {n_contracts} contracts...")

pricing_spread = 0.002
call_K_interp = np.linspace(s*(1+pricing_spread), s*(1+3*pricing_spread),int(n_strikes))
put_K_interp = np.linspace(s*(1-3*pricing_spread),s*(1-pricing_spread),int(n_strikes))


print(f"\ntrain s: {s}")
print(f"strikes between {int(min(put_K_interp))} and {int(max(call_K_interp))}")
print(f"maturities between {int(min(T))} and {int(max(T))} days")

call_features = generate_train_features(call_K_interp, T, s, ['call'])
call_features['moneyness'] = call_features['spot_price']/call_features['strike_price']

put_features = generate_train_features(put_K_interp, T, s, ['put'])
put_features['moneyness'] = put_features['strike_price']/put_features['spot_price']

train_K = np.sort(np.array([put_K_interp,call_K_interp],dtype=int).flatten())


features = call_features


features = pd.concat(
    [call_features,put_features],ignore_index=True).reset_index(drop=True)



features = features.apply(bicubic_vol_row,axis=1,bicubic_vol = ms.bicubic_vol)

features['dividend_rate'] = 0.02
features['risk_free_rate'] = 0.04

heston_parameters = calibrate_heston(features,ms.s,ms.calculation_date)

features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]



heston_features = features.apply(ms.heston_price_vanilla_row,axis=1)
ml_data = noisyfier(heston_features)



pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
print(f"\n\ntraining dataset:\n{ml_data}")
print(f"\n\ndescriptive statistics:\n{ml_data.describe()}")
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')

