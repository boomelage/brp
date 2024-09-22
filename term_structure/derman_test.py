# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:15:10 2024

"""
# =============================================================================
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(current_dir)
sys.path.append(parent_dir)
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import cm
# =============================================================================
from routine_ivol_collection import raw_calls, raw_puts

s = 5625

T =  [
            
            2, 
            
            3,   
            
            7,
            
            8,   9,  10,  
            
            14,  
            
            15,  16,  17,  21,  22,  23, 24,  
            
            28,
            
            29,  
            
            30,  
            
            31, 
            
            37,  39,  46, 
            
            60,  
            
            74, 
            
            95, 
            
            106, 
            
            158, 
            
            165, 
            
            186, 
            
            196, 242, 277, 287, 305, 
            
            368, 
            
            459, 487, 640
            
            ]
        # sep 16th

raw_calls = raw_calls
raw_puts = raw_puts
raw_call_K = raw_calls.index
raw_put_K = raw_puts.index

put_atmvols = raw_calls.loc[s,:].dropna()
call_atmvols = raw_puts.loc[s,:].dropna()
call_K = raw_calls.index[raw_calls.index>s]
put_K = raw_puts.index[raw_puts.index<s]

# =============================================================================
"""
computing Derman coefficients

"""
def compute_derman_coefficients(s,T,K,atm_volvec,raw_ts):
    
    derman_coefs_np = np.zeros(len(T),dtype=float)
    derman_coefs = pd.DataFrame(derman_coefs_np)
    derman_coefs.index = T
    
    for t in T:
        try:
            t = int(t)
            term_struct = raw_ts.loc[:,t].dropna()
            
            K_reg = term_struct.index
            x = np.array(K_reg  - s, dtype=float)
            y = np.array(term_struct - atm_volvec[t],dtype=float)
        
            model = LinearRegression(fit_intercept=False)
            x = x.reshape(-1,1)
            model.fit(x,y)
            b = model.coef_[0]
    
            derman_coefs.loc[t] = b
        except Exception:
            print(f'error: t = {t}')
    return derman_coefs


derman_coefs = compute_derman_coefficients(s, T, raw_call_K, call_atmvols, raw_calls)



"""
surface maker

"""

T = derman_coefs.index
derman_ts = pd.DataFrame(np.zeros((len(raw_call_K),len(T)),dtype=float))
derman_ts.index = raw_call_K
derman_ts.columns = T

for i, k in enumerate(raw_call_K):
    moneyness = k-s
    for j, t in enumerate(T):
        derman_ts.loc[k,t] = (
            np.array(call_atmvols,dtype=float)[j] + \
            np.array(derman_coefs)[j] * moneyness
        )
    derman_ts = derman_ts[~(derman_ts<0)].dropna(how="any",axis=0)


"""
testing approximation fit
"""


# derman_test_ts = derman_ts
# real_test_ts = raw_calls

# for t in T: 
#     actual_data = real_test_ts.loc[:,t].dropna()
#     plot_K = actual_data.index
#     derman_fit = derman_test_ts.loc[plot_K,t]
#     plt.rcParams['figure.figsize']=(6,4)
#     # K = K.astype(int)
#     fig, ax = plt.subplots()
#     ax.plot(raw_call_K, derman_ts)
#     ax.plot(raw_call_K, raw_calls, "o")
#     ax.set_title("Derman approximation of implied volatility surface")
#     plt.show()
#     plt.cla()
#     plt.clf()
    

derman_coefs