# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:57:11 2024

@author: boomelage
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
historical_data_dir = os.path.abspath(os.path.join(parent_dir,'historical_data','hist_outputs'))
spontaneous_data_directory = os.path.abspath(os.path.join(current_dir,'barriers'))


from data_query import dirdatacsv
from settings import model_settings, compute_moneyness
ms = model_settings()


DATA_DIRECTORY = historical_data_dir

os.chdir(DATA_DIRECTORY)
csvs = dirdatacsv()


training_data = pd.DataFrame()
for file in csvs:
    train_subset = pd.read_csv(file)
    training_data = pd.concat([training_data,train_subset],ignore_index=True)


training_data = training_data.drop(
    columns=training_data.columns[0]).drop_duplicates()


training_data['calculation_date'] = pd.to_datetime(training_data['calculation_date'])
training_data['expiration_date'] = pd.to_datetime(training_data['expiration_date'])

initial_count = training_data.shape[0]

"""
maturities filter
"""

# training_data = training_data[
#     (abs(training_data['days_to_maturity'])>=0)&
#     (abs(training_data['days_to_maturity'])<=31)
#     ].reset_index(drop=True)


# training_data = training_data[
#       (training_data['calculation_date']>=datetime(2007,1,3))
#       &
#       (training_data['calculation_date']<=datetime(2007,1,3))
#     ].reset_index(drop=True)


"""
type filter
"""
 
# training_data = training_data[training_data.loc[:,'barrier_type_name'] == 'DownOut']
# training_data = training_data[training_data.loc[:,'w'] == 'put']


""""""
training_data = compute_moneyness(training_data)

""""""
"""
moneyness filter
"""

# training_data = training_data[
#     (training_data['moneyness']>=-1.0)&
#     (training_data['moneyness']<=-0.0)
#     ].reset_index(drop=True)


""""""

training_data['moneyness_tag'] = ms.encode_moneyness(training_data['moneyness'])

training_data = training_data[
    [ 
     'spot_price', 'strike_price', 'barrier', 'heston_price', 'barrier_price',
     'barrier_type_name', 'days_to_maturity', 'moneyness', 'outin', 'w', 
     'updown', 'moneyness_tag','calculation_date', 'expiration_date',
     'observed_price', 'theta', 'kappa', 'rho', 'eta','v0', 
        ]
    ]

T = np.sort(training_data['days_to_maturity'].unique())
B = np.sort(training_data['barrier_type_name'].unique())
W = np.sort(training_data['w'].unique())
S = np.sort(training_data['spot_price'].unique())
K = np.sort(training_data['strike_price'].unique())
training_data = training_data.dropna()
total_contracts = training_data.shape[0]
pd.set_option("display.max_columns",None)
print(f"\n{training_data.describe()}\nspot(s):\n{S}\n\nstrikes:\n{K}\n\n"
      f"maturities:\n{T}\n\ntypes:\n{W}\n{B}\ninitial count:\n{initial_count}"
      f"\ntotal contracts:\n{total_contracts}\n")
pd.reset_option("display.max_columns")

os.chdir(parent_dir)