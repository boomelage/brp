# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:38:52 2024

"""
import numpy as np
import pandas as pd
from itertools import product
import QuantLib as ql
import math
from heston_calibration import calibrate_heston
from pricing import heston_price_vanillas, noisyfier



class bloomberg_ivols:
    def __init__(self,dividend_rate,data_files):
            self.implied_vols_matrix = None
            self.S = None
            self.K = None
            self.T = None
            self.dataset = None
            self.ivol_table = None
            self.black_var_surface = None
            self.strikes =  None
            self.maturities = None
            dividend_rate = dividend_rate
            data_files = data_files
            
    def generate_from_market_data(data_files, dividend_rate):    
        
          def clean_data():
              calls = pd.DataFrame()
              puts = pd.DataFrame()
              for file in data_files:
                  octo = pd.read_excel(f"{str(file)}")
                  octo = octo.dropna()
                  octo.columns = octo.iloc[0]
                  octo = octo.drop(index = 0).reset_index().drop(
                      columns = 'index')
                  splitter = int(octo.shape[1]/2)
                  octoputs = octo.iloc[:,:-splitter]
                  octocalls = octo.iloc[:,:splitter]
                  octocalls.loc[:,'w'] = 1
                  octoputs.loc[:,'w'] = -1
                  calls = pd.concat([calls, octocalls], ignore_index=True)
                  puts = pd.concat([puts, octoputs], ignore_index=True)
                  calls = calls.sort_values(by = 'Strike')
                  puts = puts.sort_values(by = 'Strike')
              calls['IVM'] = calls['IVM']/100
              puts['IVM'] = puts['IVM']/100
              return calls, puts
      
          def group_by_maturity(df):
              grouped = df.groupby('DyEx')
              group_arrays = []
              for _, group in grouped:
                  group_array = group.values
                  group_arrays.append(group_array)
              final_array = np.array(group_arrays, dtype=object)
              return final_array
      
          # full option data
          calls, puts = clean_data()
          # ivol table generation
          callvols = calls.copy().drop(columns = ['w','Rate'])
          groupedmat = group_by_maturity(callvols)
          n_maturities = int(len(groupedmat))
          n_strikes = int(len(groupedmat[0]))
          maturities = calls['DyEx'].unique
      
          implied_vols_matrix = ql.Matrix(n_strikes,n_maturities,float(0))
      
          for i in range(n_maturities):
              for j in range(n_strikes):
                  implied_vols_matrix[j][i] = groupedmat[i][j][1]
          S = int(np.median(calls['Strike'].unique()))
          K = calls['Strike'].unique().to_numpy()
          T = calls['DyEx'].unique.to_numpy
          
          return implied_vols_matrix, calls, puts, S, K, T
          
          
          features = calls.copy()
          features["years_to_maturity"] = features["years_to_maturity"]/365
          features['risk_free_rate'] = ['Rate']
          features['dividend_rate'] = dividend_rate
          features['w'] = 1
          option_data = features
          option_data['calculation_date'] = ql.Date.todaysDate()
          option_data['maturity_date'] = option_data.apply(
              lambda row: row['calculation_date'] + ql.Period(
                  int(math.floor(row['years_to_maturity'] * 365)), ql.Days), axis=1)
          
          spot = float(S[0])
          calculation_date = ql.Date.todaysDate()
          expiration_dates = []
          for maturity in maturities:
              expiration_date = calculation_date + ql.Period(int(maturity), ql.Days)
              expiration_dates.append(expiration_date)
          day_count = ql.Actual365Fixed()
          day_count = ql.Actual365Fixed()
          calendar = ql.UnitedStates(m=1)
          ql.Settings.instance().evaluationDate = calculation_date
          dividend_yield = ql.QuoteHandle(ql.SimpleQuote(dividend_rate))
          dividend_rate = dividend_yield
          flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(
              calculation_date, risk_free_rate, day_count))
          dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(
              calculation_date, dividend_rate, day_count))
          
          black_var_surface = ql.BlackVarianceSurface(
              calculation_date, calendar,
              expiration_dates, K,
              implied_vols_matrix, day_count)
          
          heston_params = calibrate_heston(
              option_data,flat_ts,dividend_ts, spot, expiration_dates, 
              black_var_surface,strikes, day_count,calculation_date, calendar, 
              dividend_rate, implied_vols_matrix)
          
          heston_vanillas = heston_price_vanillas(heston_params)
          
          dataset = noisyfier(heston_vanillas)
      
          return dataset, ivol_table, implied_vols_matrix, \
              black_var_surface, strikes, maturities
      
  
  
  
  
  





