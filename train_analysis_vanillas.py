# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:16:55 2024

@author: boomelage
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
sys.path.append('historical_data')

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


from train_main import train_data, train_X, train_y, \
    test_data, test_X, test_y, model_fit, runtime

def compute_performance():
    training_results = test_X.copy()
    training_results.loc[
        training_results['w'] == 'call', 'moneyness'
        ] = training_results['spot_price'] / training_results['strike_price'] - 1
    training_results.loc[
        training_results['w'] == 'put', 'moneyness'
        ] = training_results['strike_price'] / training_results['spot_price'] - 1
    training_results['target'] = test_y
    training_results['prediciton'] = model_fit.predict(test_X)
    training_results['absRelError'] = abs(training_results['prediciton']/training_results['target']-1)

# plt.figure()
# plt.scatter(training_results['moneyness'],training_results['absRelError'])
# plt.xlabel('percentage moneyness')
# plt.ylabel('absolute relative error')
# plt.title('prediciton error')

# print(f"\n{training_results}\n")
# pd.reset_option("display.max_rows")
# pd.reset_option("display.max_columns")