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
ms = model_settings()
from tqdm import tqdm
import numpy as np
import QuantLib as ql
from routine_calibration_global import heston_parameters


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

K = np.linspace(s*0.9,s*1.1,4000)

T = ms.T

T = np.arange(min(T),max(T),1)

print(f"\ngenerating {2*len(K)*len(T)} contracts...\n")

features = generate_train_features(K, T, s, ['call','put'])

features['dividend_rate'] = 0.02
features['risk_free_rate'] = 0.04
features['sigma'] = heston_parameters['sigma'].iloc[0]
features['theta'] = heston_parameters['theta'].iloc[0]
features['kappa'] = heston_parameters['kappa'].iloc[0]
features['rho'] = heston_parameters['rho'].iloc[0]
features['v0'] = heston_parameters['v0'].iloc[0]
features['heston_price'] = 0.00

progress_bar = tqdm(desc="pricing",total=features.shape[0],unit= "contracts")

for i, row in features.iterrows():
    s = row['spot_price']
    k = row['strike_price']
    t = int(row['days_to_maturity'])
    r = row['risk_free_rate']
    g = row['dividend_rate']
    v0 = row['v0']
    kappa = row['kappa']
    theta = row['theta']
    sigma = row['sigma']
    rho = row['rho']
    w = row['w']
    
    
    date = ms.calculation_date + ql.Period(t,ql.Days)
    option_type = ql.Option.Call if w == 'call' else ql.Option.Put
    
    payoff = ql.PlainVanillaPayoff(option_type, k)
    exercise = ql.EuropeanExercise(date)
    european_option = ql.VanillaOption(payoff, exercise)
    flat_ts = ms.make_ts_object(r)
    dividend_ts = ms.make_ts_object(g)
    
    heston_process = ql.HestonProcess(
        flat_ts,dividend_ts, 
        ql.QuoteHandle(ql.SimpleQuote(s)), 
        v0, kappa, theta, sigma, rho)
    
    heston_model = ql.HestonModel(heston_process)
    
    engine = ql.AnalyticHestonEngine(heston_model)
    
    european_option.setPricingEngine(engine)
    
    h_price = european_option.NPV()
    progress_bar.update(1)
    features.at[i, 'heston_price'] = h_price
    
progress_bar.close()

ml_data = noisyfier(features)

print(f"\n{ml_data}\n{ml_data.describe()}\n")

