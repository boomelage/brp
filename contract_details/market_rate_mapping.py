# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:44:54 2024

@author: boomelage
"""
import numpy as np
import pandas as pd
from routine_collection import contract_details

"""
                                getting r and g pivots from current market_data
"""

details_indexed = contract_details.copy().set_index([
    'strike_price','days_to_maturity'])
rfrpivot = contract_details.pivot_table(
    values = 'risk_free_rate', 
    index = 'strike_price', 
    columns = 'days_to_maturity'
    )
dvypivot = contract_details.pivot_table(
    values = 'dividend_rate', 
    index = 'strike_price', 
    columns = 'days_to_maturity'
    )

rfrpivot = rfrpivot.interpolate(method='linear', axis=0)
rfrpivot = rfrpivot.dropna(how='any',axis=0).dropna(how='any',axis=1)

dvypivot = dvypivot.interpolate(method='linear', axis=0)
dvypivot = dvypivot.dropna(how='any',axis=0).dropna(how='any',axis=1)


rvec = np.zeros(rfrpivot.shape[1],dtype=float)
rvec = pd.DataFrame(rvec)
rvec.index = rfrpivot.columns
for i, k in enumerate(rfrpivot.index):
    for j, t in enumerate(rfrpivot.columns):
        rvec.loc[t] = float(np.median(rfrpivot.loc[:, t].dropna().unique()))

gvec = np.zeros(dvypivot.shape[1],dtype=float)
gvec = pd.DataFrame(gvec)
gvec.index = dvypivot.columns
for i, k in enumerate(dvypivot.index):
    for j, t in enumerate(dvypivot.columns):
        gvec.loc[t] = float(np.median(dvypivot.loc[:, t].dropna().unique()))

rates_dict = {'risk_free_rate':rvec,'dividend_rate':gvec}

    # example
t = (rvec.index[45],0)
rt0 = rates_dict['risk_free_rate'].loc[t]
print(f'\nexample rt0: {rt0}\n')



"""
mapping appropriate rates
"""

def map_rate(ratename,features):
    for row in features.index:
        try:
            t = (int(features.iloc[row]['days_to_maturity']),0)
            features.loc[row,ratename] = rates_dict[ratename].loc[t]
        except Exception as e:
            print(e)
    return features
        

# features = map_rate('risk_free_rate')
# features = map_rate('dividend_rate')