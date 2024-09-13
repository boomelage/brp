#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:40:38 2024

This class collects market data exported from the 'calls/puts' tab in OMON

below is a routine which can take a dataset of spots, strikes, atm implied
volatility, risk_free_rate, dividend_yield, maturities, and momentarily a
static flag 'w' set to 1 indicating a call payoff

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
import numpy as np
import QuantLib as ql
from data_query import dirdata
from pricing import BS_price_vanillas, heston_price_vanillas, noisyfier
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')

class routine_collection():
    def __init__(self):
        self.data_files = dirdata()
        self.market_data = pd.DataFrame()
        self.excluded_file = None
        
    def collect_call_data(self,file):
        df = pd.read_excel(file)
        df = df.dropna().reset_index(drop=True)
        df.columns = df.loc[0]
        df = df.iloc[1:,:]
        df = df.astype(float)
        splitter = int(len(df.columns)/2)
        calls = df.iloc[:,:splitter]
        calls = calls[~(calls['DyEx'] < 1)]
        calls['spot_price'] = np.median(calls['Strike'])
        calls['volatility'] = calls['IVM'] / 100
        calls['dividend_rate'] = calls['DvYd'] / 100
        calls['risk_free_rate'] = calls['Rate'] / 100
        calls['days_to_maturity'] = calls['DyEx'].astype(int)
        calls['strike_price'] = calls['Strike'].astype(int)
        calls['w'] = 1
        calls = calls.drop(columns = ['IVM','DvYd','Rate','DyEx','Strike'])

        print(f"\nfile: {file}")
        print(calls['days_to_maturity'].unique())
        print(f"maturities count: {len(calls['days_to_maturity'].unique())}")
        print(calls['strike_price'].unique())
        print(f"strikes count: {len(calls['strike_price'].unique())}")

        return calls

    def concat_option_data(self):
        market_data = pd.DataFrame()
        for file in self.data_files:
            df = self.collect_call_data(file)
            market_data = pd.concat([market_data, df], ignore_index=True)
            market_data = market_data.sort_values(by='days_to_maturity')
            market_data = market_data.reset_index(drop=True)
        return market_data

    def collect_market_data_and_price(self):
        market_data = self.concat_option_data()
        
        calculation_date = ql.Date.todaysDate()
        from routine_calibration import heston_params
        market_data['v0'] = heston_params['v0']
        market_data['kappa'] = heston_params['kappa']
        market_data['theta'] = heston_params['theta']
        market_data['sigma'] = heston_params['sigma']
        market_data['rho'] = heston_params['rho']
        market_data['calculation_date'] = calculation_date
        def compute_maturity_date(row):
            row['maturity_date'] = calculation_date + ql.Period(
                int(row['days_to_maturity']),ql.Days)
            return row
        
        market_data = market_data.apply(compute_maturity_date, axis=1)
        
        option_prices = BS_price_vanillas(market_data)
        option_prices = heston_price_vanillas(option_prices)
        option_prices = noisyfier(option_prices)
        priced_market_data = option_prices.dropna()
        print(priced_market_data)
        return priced_market_data
    
    def collect_market_data(self):
        market_data = self.concat_option_data()
        
        calculation_date = ql.Date.todaysDate()
        
        market_data['calculation_date'] = calculation_date
        
        def compute_maturity_date(row):
            row['maturity_date'] = calculation_date + ql.Period(
                int(row['days_to_maturity']),ql.Days)
            return row
        
        market_data = market_data.apply(compute_maturity_date, axis=1)
        print(market_data)
        return market_data


"""
routine which can take a dataset of spots, strikes, atm implied
volatility, risk_free_rate, dividend_yield, maturities, and momentarily a
static flag 'w' set to 1 indicating a call payoff

"""

# =============================================================================
                                                              # data collection

rc = routine_collection()
try:
    contract_details = rc.collect_market_data()
except Exception:
    print('check working directory files!')
    
contract_details = contract_details.copy()
contract_details['atm_vol'] = 0.1312


K = contract_details['strike_price'].unique()
T = contract_details['days_to_maturity'].unique()

# =============================================================================
                                                                       # Derman
from Derman import retrieve_derman_from_csv, derman
derman_coefs, derman_maturities = retrieve_derman_from_csv()
derman = derman(derman_coefs = derman_coefs)

contract_details = contract_details[
    contract_details['days_to_maturity'].isin(derman_maturities)]

contract_details = contract_details.reset_index(drop=True)


def apply_derman_vols_row(row):
    s = row['spot_price']
    k = row['strike_price']
    t = row['days_to_maturity']
    atm_vol = row['atm_vol']
    
    if t not in derman_coefs.columns:
        print(f"Days to maturity {t} not found in derman_coefs. Skipping row.")
        row['volatility'] = np.nan
        return row
    
    try:
        derman_vol = derman.compute_one_derman_vol(s, k, t, atm_vol)
        row['volatility'] = derman_vol
    except Exception as e:
        print(f"Error computing Derman vol for row: {e}")
        row['volatility'] = np.nan
    
    return row

contract_details = contract_details.apply(
    apply_derman_vols_row, axis=1).dropna(
        subset=['volatility']).reset_index(drop=True)

# def apply_derman_vols_row(row):
#     s = row['spot_price']
#     k = row['strike_price']
#     t = row['days_to_maturity']
#     atm_vol = row['atm_vol']
#     derman_vol = derman.compute_one_derman_vol(s, k, t, atm_vol)
#     row['volatility'] = derman_vol
#     return row


# contract_details = contract_details.apply(apply_derman_vols_row,axis=1)











