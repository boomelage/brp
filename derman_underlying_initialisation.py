#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:23:06 2024

"""

from Derman import derman

derman = derman()

"""
# =============================================================================
                   # loading Derman coefficients from OMON term strtucture data
"""

# derman_coefs = derman.get_derman_coefs()
# derman_coefs.to_csv(r'derman_coefs.csv')

"""
# =============================================================================
                                         # loading Derman coefficients from csv
"""

from Derman import retrieve_derman_from_csv
derman_coefs, derman_maturities = retrieve_derman_from_csv()

"""
# =============================================================================
                                                 # loading option data from csv
"""

# import pandas as pd
# contract_details = pd.read_csv(r'test_contract_details.csv')
# contract_details.index = contract_details[contract_details.columns[0]]
# contract_details = contract_details.drop(
#     columns = contract_details.columns[0]).reset_index(drop=True)

"""
# =============================================================================
                                                                  # from market
"""


from routine_collection import contract_details
s = contract_details['spot_price'].unique()[0]
K = contract_details['strike_price'].unique()
T = contract_details['days_to_maturity'].unique()


from routine_ivol_collection import implied_vols
atm_vol_df = implied_vols.loc[s]

derman_df = derman.make_derman_df(s,K,T,atm_vol_df)

derman_df