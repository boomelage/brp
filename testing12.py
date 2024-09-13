# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:06:09 2024

@author: boomelage
"""

from derman_underlying_initialisation import derman_coefs, derman_maturities,\
    implied_vols, contract_details, S, K, T, atm_vol_df
    
from Derman import derman

import numpy as np

derman = derman(derman_coefs=derman_coefs,implied_vols=implied_vols)


derman


s = S[1]
atm_vol = [0.1312]
derman_atm_dfs = np.empty(len(S),dtype=object)
for i, s in enumerate(S):
    
    derman_df_for_s = derman.make_derman_df_for_S(
        s, K, T, atm_vol_df, contract_details, derman_coefs, derman_maturities)
    derman_df_for_s = derman_df_for_s.loc[min(S):max(S)]
    derman_atm_dfs[i] = derman_df_for_s
derman_atm_dfs[2]   

