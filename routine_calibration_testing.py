# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:58:27 2024

"""

import QuantLib as ql
import pandas as pd
import numpy as np
from settings import model_settings
ms = model_settings()



def test_heston_calibration(
        calibration_dataset, heston_parameters,calculation_date,r,g
        ):
    
    test_dataset = calibration_dataset.copy()
    for i, row in test_dataset.iterrows():
        s = row['spot_price']
        k = row['strike_price']
        moneyness = k-s
        if moneyness < 0:
            test_dataset.at[i,'w'] = 'put'
        else:
            test_dataset.at[i,'w'] = 'call'
    
    test_dataset['kappa'] = heston_parameters['kappa']
    test_dataset['theta'] = heston_parameters['theta']
    test_dataset['rho'] = heston_parameters['rho']
    test_dataset['eta'] = heston_parameters['eta']
    test_dataset['v0'] = heston_parameters['v0']
    test_dataset['calculation_date'] = calculation_date
    test_dataset['black_scholes'] = ms.vector_black_scholes(test_dataset)
    
    test_dataset['ql_heston_price'] = ms.vector_heston_price(test_dataset)
    
    print_test = test_dataset[
        ['w', 'spot_price','strike_price', 'days_to_maturity', 
          'ql_heston_price',
        'black_scholes']].copy()
    
    print_test['relative_error'] = \
          (print_test['ql_heston_price']/print_test['black_scholes'])-1
         
    test_avg = np.average(np.abs(np.array(print_test['relative_error'])))
    test_avg_print = f"{round(test_avg*100,4)}%"
    
    heston_parameters['relative_error'] = test_avg
    
    pd.set_option("display.max_columns",None)
    print(f"\ncalibration test:\n{print_test}\n"
          f"repricing average absolute relative error: {test_avg_print}"
          f"\n{heston_parameters}\n")
    pd.reset_option("display.max_columns")
    
    return heston_parameters


