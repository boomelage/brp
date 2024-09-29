# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:10:31 2024

generation routine

"""
import os
import sys
import time
import pandas as pd
import numpy as np
import QuantLib as ql
# from tqdm import tqdm
from itertools import product
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,'train_data'))
sys.path.append(os.path.join(parent_dir,'term_structure'))
vanilla_csv_dir = os.path.join(current_dir,'historical_vanillas')
import Derman as derman
from routine_calibration_global import calibrate_heston
from routine_calibration_testing import test_heston_calibration
# from train_generation_barriers import generate_barrier_features, \
#     generate_barrier_options
from settings import model_settings
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import historical_data


"""
# =============================================================================
                        historical generation routine
"""

r = 0.04
historical_calibration_errors = pd.Series()
calibration_errors = []
for row_i, row in historical_data.iterrows():
# row = historical_data.iloc[0]


    
    s = row['spot_price']
    g = row['dividend_rate']/100
    dtdate = row['date']
    print_date = dtdate.strftime('%A %d %B %Y')
    calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)
    
    ###############
    # CALIBRATION #
    ###############
                    
    ql_T = [
         ql.Period(30,ql.Days),
         ql.Period(60,ql.Days),
         ql.Period(3,ql.Months),
         ql.Period(6,ql.Months),
         ql.Period(12,ql.Months),
         ql.Period(18,ql.Months),
         ql.Period(24,ql.Months)
         ]
    
    expiration_dates = []
    for t in ql_T:
        expiration_dates.append(calculation_date + t)
    T = []
    for date in expiration_dates:
        T.append(date - calculation_date)
    
    T = [30,60,95,186,368]
    
    atm_volvec = np.array(row[
        [
            '30D', '60D', '3M', '6M', '12M', 
            ]
        ]/100,dtype=float)
    
    atm_volvec = pd.Series(atm_volvec)
    atm_volvec.index = T
    
    
    spread = 0.05
    initial_spread = 0.005
    wing_size = 3
    
    put_K = np.linspace(s*(1-spread),s*0.995,wing_size)
    
    call_K = np.linspace(s*1.005,s*(1+spread),wing_size)
    
    K = np.unique(np.array([put_K,call_K]).flatten())
    
    T = [30,60,95]
    calibration_dataset =  pd.DataFrame(
        product(
            [s],
            K,
            T,
            ),
        columns=[
            'spot_price', 
            'strike_price',
            'days_to_maturity',
                  ])
    
    
    calibration_dataset['volatility'] = ms.derman_volatilities(
        s, 
        calibration_dataset['strike_price'], 
        calibration_dataset['strike_price'], 
        calibration_dataset['days_to_maturity'].map(derman.derman_coefs), 
        calibration_dataset['days_to_maturity'].map(atm_volvec)
        )
    
    heston_parameters = calibrate_heston(
        calibration_dataset, s, r, g, calculation_date)
    
    heston_parameters = test_heston_calibration(
        calibration_dataset, heston_parameters, calculation_date, r, g)
    print(f"^ {print_date}")
    calibration_error = heston_parameters['relative_error']
    historical_calibration_errors[dtdate] = calibration_error
    ###################
    # DATA GENERATION #
    ###################
                    
                    
    # if abs(calibration_error) <= 0.2:
        
    #     T = np.arange(7,365,7)
        
    #     K = np.linspace(s*0.8,s*1.2,120)
        
    #     W = ['call','put']
        
        
        
    #     ############
    #     # VANILLAS #
    #     ############
        
        
    #     vanilla_features = pd.DataFrame(
    #         product(
    #             [float(s)],
    #             K,
    #             T,
    #             W
    #             ),
    #         columns=[
    #             "spot_price", 
    #             "strike_price",
    #             "days_to_maturity",
    #             "w"
    #                   ])
        
        
    #     expiration_dates = ms.vexpiration_datef(
    #         vanilla_features['days_to_maturity'],calculation_date)
        
    #     vanilla_features['calculation_date'] = calculation_date
    #     vanilla_features['expiration_date'] = expiration_dates
        
    #     hestons = ms.vector_heston_price(
    #             vanilla_features['spot_price'],
    #             vanilla_features['strike_price'],
    #             r,g,
    #             vanilla_features['w'],
    #             heston_parameters['v0'],
    #             heston_parameters['kappa'],
    #             heston_parameters['theta'],
    #             heston_parameters['eta'],
    #             heston_parameters['rho'],
    #             calculation_date,
    #             vanilla_features['expiration_date']
    #         )
    #     vanilla_features['heston_price'] = hestons
        
    #     historical_contracts = pd.concat(
    #         [historical_contracts, vanilla_features],
    #         ignore_index = True
    #         )
        
    #     os.path.join(current_dir,)
    #     file_time = datetime.fromtimestamp(time.time())
    #     file_tag = file_time.strftime("%Y-%m-%d %H%M%S")
    #     file_name = dtdate.strftime("%Y-%m-%d") \
    #         + f" spot{int(s)} " + file_tag + r".csv"
    #     historical_contracts.to_csv(os.path.join(vanilla_csv_dir,file_name))
    
    #     ############
    #     # BARRIERS #
    #     ############
    
    # # =============================================================================
    # #     # """
    # #     # # up_barriers = np.linspace(s*1.01,s*1.19,50)
    # #     # # down_barriers = np.linspace(s*0.81,s*0.99,50)
    # #     
    # #     # # down_features = generate_barrier_features(
    # #     # #     s,K,T,down_barriers,'Down', ['Out','In'], ['call','put']
    # #     # #     )
    # #     
    # #     # # up_features = generate_barrier_features(
    # #     # #     s,K,T,up_barriers,'Up', ['Out','In'], ['call','put']
    # #     # #     )
    # #     
    # #     # # features = pd.concat([down_features,up_features],ignore_index=True)
    # #     # # features['barrier_type_name'] = features['updown'] + features['outin']
    # #     
    # #     # # barrier_options = generate_barrier_options(
    # #     # #     features,calculation_date,heston_parameters, g, r'hist_outputs')
    # #     
    # #     # # historical_contracts = pd.concat(
    # #     # #     [historical_contracts, barrier_options],ignore_index=True)
    # #     # """
    # # =============================================================================
        
    #     print(f"\n{historical_contracts.describe()}\n{print_date}")
    # else:
    #     test_large_error = str(f"### large calibration error: "
    #                             f"{round(calibration_error*100,2)}% ###")
    #     print('#'*len(test_large_error))
    #     print('#'*len(test_large_error))
    #     print('#'*len(test_large_error))
    #     print(test_large_error)
    #     print('#'*len(test_large_error))
    #     print('#'*len(test_large_error))
    #     print('#'*len(test_large_error))
    #     print(print_date)
        
        
        
        
            
            
            
            
            