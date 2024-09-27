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

derman_s = 5625

derman_T =  [
            
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

put_atmvols = raw_calls.loc[derman_s,:].dropna()
call_atmvols = raw_puts.loc[derman_s,:].dropna()
call_K = raw_calls.index[raw_calls.index>derman_s]
put_K = raw_puts.index[raw_puts.index<derman_s]

# =============================================================================
"""
computing Derman coefficients

"""
def compute_derman_coefficients(s,T,K,atm_volvec,raw_ts):
    
    derman_coefs = pd.Series(np.zeros(len(T),dtype=float),index=T)
    for t in T:
        try:
            t = int(t)
            term_struct = raw_ts.loc[:,t].copy()
            term_struct = term_struct.replace(0,np.nan).dropna()
            
            K_reg = term_struct.index
            x = np.array(K_reg  - s, dtype=float)
            y = np.array(term_struct - atm_volvec[t],dtype=float)
        
            model = LinearRegression(fit_intercept=False)
            x = x.reshape(-1,1)
            model.fit(x,y)
            b = model.coef_[0]
            derman_coefs[t] = b
        
        except Exception:
            print(f'Derman error: t = {t}')
    return derman_coefs


derman_coefs = compute_derman_coefficients(
    derman_s, derman_T, raw_call_K, call_atmvols, raw_calls)


# """
# surface maker

# """

# def make_derman_suface(s,K,T,derman_coefs,atm_volvec):
#     derman_ts = pd.DataFrame(np.zeros((len(K),len(T)),dtype=float))
#     derman_ts.index = K
#     derman_ts.columns = T
#     for k in K:
#         for t in T:
#             moneyness = k-s
#             derman_ts.loc[k,t] = atm_volvec[t] + derman_coefs[t]*moneyness
#     return derman_ts
            
            


# """
# plotting approximation fit
# """ 


# for t in T: 
#     actual_data = raw_puts.loc[:,t].replace(0,np.nan).dropna()
#     derman_fit = derman_ts.loc[actual_data.index,t]
#     plt.rcParams['figure.figsize']=(6,4)
#     fig, ax = plt.subplots()
    
#     ax.plot(actual_data.index, derman_fit)
#     ax.plot(actual_data.index, actual_data, "o")
    
#     ax.set_title("Derman approximation of implied volatility surface")
    
#     plt.show()
#     plt.cla()
#     plt.clf()
