# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:10:31 2024

generation routine
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join('term_structure',parent_dir))
import pandas as pd
import numpy as np
import QuantLib as ql
import time
from datetime import datetime
from itertools import product
from routine_calibration_global import calibrate_heston
from bicubic_interpolation import make_bicubic_functional, bicubic_vol_row
from pricing import noisyfier
from settings import model_settings
from tqdm import tqdm
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import collect_historical_data



def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity"
                  ])
    return features



"""
historical generation routine
"""
historical_data = collect_historical_data()

total = historical_data.shape[0]
historical_option_data = pd.DataFrame()
progress_bar = tqdm(
    desc="generating",total = total,leave=True,unit='days',dynamic_ncols=True)

for i, row in historical_data.iterrows():
    s = row['spot_price']
    dtdate = row['date']
    calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)
    expiry_dates = np.array([
            calculation_date + ql.Period(30,ql.Days), 
            calculation_date + ql.Period(60,ql.Days), 
            calculation_date + ql.Period(3,ql.Months), 
            calculation_date + ql.Period(6,ql.Months),
            calculation_date + ql.Period(12,ql.Months), 
            # calculation_date + ql.Period(18,ql.Months), 
            # calculation_date + ql.Period(24,ql.Months)
          ],dtype=object)
    
    T = expiry_dates - calculation_date
    
    """
    calibration dataset construction
    """


    n_hist_spreads = 5
    historical_spread = 0.005
    n_strikes = 3
    K = np.linspace(
        s*(1 - n_hist_spreads * historical_spread),
        s*(1 + n_hist_spreads * historical_spread),
        n_strikes)
    
    bicubic_vol = make_bicubic_functional(
        ms.derman_ts, 
        ms.derman_ts.index.tolist(), 
        ms.derman_ts.columns.tolist())
    
    calibration_dataset = generate_features(
        K, T, s)

    calibration_dataset = calibration_dataset.apply(
        bicubic_vol_row, axis = 1, bicubic_vol = bicubic_vol)
    calibration_dataset = calibration_dataset.copy()
    calibration_dataset['risk_free_rate'] = 0.04
    calibration_dataset['dividend_rate'] = row['dividend_rate']
    
    heston_parameters, performance_df = calibrate_heston(
        calibration_dataset, s, calculation_date)
    
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T[:3],
            ['call','put']
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity",
            "w"
                  ])
    
    features['sigma'] = heston_parameters['sigma'].iloc[0]
    features['theta'] = heston_parameters['theta'].iloc[0]
    features['kappa'] = heston_parameters['kappa'].iloc[0]
    features['rho'] = heston_parameters['rho'].iloc[0]
    features['v0'] = heston_parameters['v0'].iloc[0]
    features['avgAbsRelErr'] = heston_parameters['avgAbsRelErr'].iloc[0]
    features['risk_free_rate'] = 0.04
    features['dividend_rate'] = row['dividend_rate']
    features['days_to_maturity'] = features['days_to_maturity'].astype(int)
    
    """
    group by t and then price to only initialise the heston process once per
    maturity per day
    """
    training_data = pd.DataFrame()
    for i, row in features.iterrows():
        s = row['spot_price']
        k = row['strike_price']
        t = int(row['days_to_maturity'])
        r = row['risk_free_rate']
        g = row['dividend_rate']
        v0 = row['v0']
        kappa = row['kappa']
        theta = row['theta']
        sigma = row['sigma']
        rho = row['rho']
        w = row['w']
        
        date = ms.calculation_date + ql.Period(t,ql.Days)
        option_type = ql.Option.Call if w == 'call' else ql.Option.Put
        
        payoff = ql.PlainVanillaPayoff(option_type, k)
        exercise = ql.EuropeanExercise(date)
        
        european_option = ql.VanillaOption(payoff, exercise)
        
        flat_ts = ms.make_ts_object(r)
        dividend_ts = ms.make_ts_object(g)
        
        heston_process = ql.HestonProcess(
            flat_ts, dividend_ts, 
            ql.QuoteHandle(ql.SimpleQuote(s)), 
            v0, kappa, theta, sigma, rho)
        
        heston_model = ql.HestonModel(heston_process)
        
        engine = ql.AnalyticHestonEngine(heston_model)
        
        european_option.setPricingEngine(engine)
        
        h_price = european_option.NPV()
        features.at[i, 'heston_price'] = h_price
        
        training_data = pd.concat([training_data, features],ignore_index=True)
        
        print(f"\n\n\n{training_data}\n\n\n")
    progress_bar.update(1)
        


progress_bar.close()


# # pd.set_option("display.max_rows",None)
# # pd.set_option("display.max_columns",None)
# # print(f"\ntraining data:\n{historical_option_data.describe()}")
# # pd.reset_option("display.max_rows")
# # pd.reset_option("display.max_columns")
# file_time = time.time()
# file_dt = datetime.fromtimestamp(file_time)
# file_timetag = file_dt.strftime("%Y-%m-%d %H-%M-%S")
# historical_option_data.to_csv(f"hist_outputs/{file_timetag}.csv")
        
