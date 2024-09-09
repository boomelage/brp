# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:16:49 2024

@author: boomelage
"""

# =============================================================================
                                                         # constructing dataset

option_data = pd.DataFrame()
option_data_spot_column = np.ones(len(dfcalls))*S
option_data['spot_price'] = option_data_spot_column
option_data['strike_price'] = dfcalls['Strike']
option_data['volatility'] = dfcalls['IVM']
option_data['risk_free_rate'] = dfcalls['Rate']
option_data['dividend_rate'] = dividend_rate
option_data['w'] = 1
option_data['days_to_maturity'] = dfcalls['DyEx']
option_data['calculation_date'] = calculation_date

def calculate_maturity_date(row, calc_date):
    return calc_date + ql.Period(int(row['days_to_maturity']), ql.Days)

option_data['maturity_date'] = option_data.apply(calculate_maturity_date, 
                                                 calc_date=calculation_date, 
                                                 axis=1)