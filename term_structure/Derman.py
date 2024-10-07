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

"""
surface maker

"""

def make_derman_suface(s,K,T,derman_coefs,atm_volvec):
    K = np.array(T.tolist())
    T =  np.array(K.tolist())
    derman_ts = pd.DataFrame(np.zeros((len(K),len(T)),dtype=float))
    for k in K:
        for t in T:
            moneyness = k-s
            derman_ts.loc[k,t] = atm_volvec[t] + derman_coefs[t]*moneyness
    return derman_ts

"""
computing Derman coefficients

"""
def compute_derman_coefficients(s,T,K,atm_volvec,raw_ts):
    derman_coefs = pd.Series(np.zeros(len(T),dtype=float),index=T)
    for i, t in enumerate(T):
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
            if b < 0: 
                derman_coefs[t] = b
            else:
                print(f"check t = {t}, b = {b}")
                derman_coefs = derman_coefs.drop(t)
        except Exception:
            pass         
    derman_coefs = derman_coefs.replace(0,np.nan).dropna()

    return derman_coefs
            
"""
plotting approximation fit
""" 

def plot_derman_test(derman_s,test_T,derman_atm_vols,raw_ts_df,derman_coefs):
    for t in test_T: 
        actual_data = raw_ts_df.loc[:,t].replace(0,np.nan).dropna()
        test_K = actual_data.index
        derman_fit = np.array(
            (derman_atm_vols[t] + derman_coefs[t]*(test_K - derman_s)),
            dtype=float
            )
        plt.rcParams['figure.figsize']=(6,4)
        fig, ax = plt.subplots()
        ax.plot(actual_data.index, derman_fit)
        ax.plot(actual_data.index, actual_data, "o")
        ax.set_title(f"Derman approximation for {int(t)} day maturity")
        plt.show()
        plt.cla()
        plt.clf()



"""
# =============================================================================
example routine

"""
from routine_ivol_collection import raw_puts

derman_s = 5625  #16th September 2024 SPX

raw_ts = raw_puts
derman_K = raw_puts.index[raw_puts.index<derman_s]
derman_T = raw_ts.columns
raw_atm_vols = pd.Series(raw_ts.loc[derman_s,:].tolist(),index=derman_T)

derman_coefs = compute_derman_coefficients(
    derman_s, derman_T, derman_K, raw_atm_vols, raw_ts)


test_T = derman_coefs.index
test_atm_vols = raw_atm_vols.loc[test_T]

# plot_derman_test(derman_s,test_T,test_atm_vols,raw_ts,derman_coefs)
# print(f"\navailable Derman coefficients:\n{derman_coefs}")





