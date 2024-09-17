# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:28:08 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
import pandas as pd
from settings import model_settings
ms = model_settings()
pd.set_option('display.max_columns',None)


import numpy as np
from routine_calibration import heston_parameters
# T = heston_parameters.index

def compute_bs_ivols(T):
    ivol_col = np.empty(len(T),dtype=float)
    for i, t in enumerate(T):
        t = t
        # S = heston_parameters.loc[t,'spot_price']
        S = ms.s
        sigma = heston_parameters.loc[t,'volatility']
        r = 0.05
        K = ms.s
        d1 = (np.log(S/K)+(r+sigma**2/2)*t/365)/(sigma*np.sqrt(t/365))
        a = t/2*d1
        b = -np.sqrt(t)
        c = np.log(S/K)+r*t/d1
        coefficients = [a,b,c]
        ivols = np.roots(coefficients)
        ivol_col[i] = ivols[1]
    heston_parameters['ivol'] = ivol_col
    return heston_parameters

# heston_parameters = compute_bs_ivols(T)
# heston_parameters['vol_err'] = heston_parameters['ivol']/heston_parameters['volatility']-1
# heston_parameters = heston_parameters[~(abs(heston_parameters['vol_err'])>0.1)]



from routine_calibration_makret import all_heston_parameters

for row_idx, row in all_heston_parameters.iterrows():
    t = row['days_to_maturity']
    S = row['spot_price']
    sigma = row[t,'volatility']
    r = 0.05
    K = ms.s
    d1 = (np.log(S/K)+(r+sigma**2/2)*t/365)/(sigma*np.sqrt(t/365))
    a = t/2*d1
    b = -np.sqrt(t)
    c = np.log(S/K)+r*t/d1
    coefficients = [a,b,c]
    ivols = np.roots(coefficients)
    heston_ivol