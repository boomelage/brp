#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:23:06 2024

@author: doomd
"""

from Derman import derman
from data_query import dirdatacsv

derman = derman()



"""
# =============================================================================
                   # loading Derman coefficients from OMON term strtucture data
"""
# from routine_collection import contract_details
# ks, mats, ts = derman.retrieve_ts()
# derman_coefs = derman.get_derman_coefs()
# derman_coefs.to_csv(r'derman_coefs.csv')
# contract_details.to_csv(r'test_contract_details.csv')


"""
# =============================================================================
                                         # loading Derman coefficients from csv
"""
derman_coefs, derman_maturities = derman.retrieve_derman_from_csv()



"""
# =============================================================================
                                                   loading option data from csv
"""
import pandas as pd
contract_details = pd.read_csv(r'test_contract_details.csv')
contract_details.index = contract_details[contract_details.columns[0]]
contract_details = contract_details.drop(
    columns = contract_details.columns[0]).reset_index(drop=True)
contract_details



# derman_df = derman.derman_ivols_for_market(market_ts,derman_coefs)

# derman_df.to_csv(r'derman_volatiltiy_surface.csv')
