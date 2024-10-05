# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:28:58 2024

"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import joblib
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
from settings import model_settings
ms = model_settings()

model_fit = joblib.load(r'deep_neural_network 2024-10-03 175823 ser169.pkl')

data = pd.read_csv(
    r'E:/git/machine-learning-option-pricing/historical_data'
    r'/historical_generation/SPX2007-2012_calibrated.csv'
    ).iloc[:,1:]


row = data.iloc[:1,:]

s = row['spot_price'].iloc[0]
K = np.linspace(s*0.8,s*1.2,10)
T = [
     30,60,90,
     # 180,360,720
     ]
r = 0.04
g = 0.02
B = np.linspace(s*0.50,s*0.99,10)
W = ['call','put']
OUTIN = ['Out','In']

test_df = pd.DataFrame(
    product(
        [s],K,T,B,[r],[g],OUTIN,['Down'],W
        ),
    columns = [
        'spot_price',
        'strike_price',
        'days_to_maturity',
        'barrier',
        'risk_free_rate',
        'dividend_rate',
        'outin',
        'updown',
        'w'
        ]
    )

test_df.loc[:,'barrier_type_name'] = test_df.loc[:,
    'updown'] + test_df.loc[:,'outin']


param_names = ['eta', 'v0', 'theta', 'kappa', 'rho']

values = row[param_names].values.flatten()

test_df[param_names] = np.tile(values, (test_df.shape[0], 1))

test_df['rebate'] = 0.00
test_df['calculation_date'] = ql.Date.todaysDate()


test_df['prediction'] = model_fit.predict(test_df)
test_df = test_df[test_df['prediction']>0]

test_df['barrier_price'] = ms.vector_barrier_price(test_df)

test_df['diff'] = test_df['prediction']-test_df['barrier_price']
test_df['sq_diff'] = test_df['diff']**2
test_df['abs_diff'] = np.abs(test_df['diff'])

RSME = np.sqrt(np.average(test_df['sq_diff']))
MAE = np.average(test_df['abs_diff'])


pd.set_option("display.max_columns",None)

print(f"\n{test_df}\nRSME: {RSME}\nMAE: {MAE}")