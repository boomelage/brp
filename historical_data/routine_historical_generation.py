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
import matplotlib.pyplot as plt
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

for row_i, row in historical_data.iterrows():
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
         # ql.Period(18,ql.Months),
         # ql.Period(24,ql.Months)
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
      
    if dtdate > datetime(2008,10,1) and dtdate < datetime(2009,4,1):
        spread = 0.3
    else:
        spread = 0.05
    
    # if s < 1000:
    #     spread = 0.15
    # elif s < 1200:
    #     spread = 0.1
    # else:
    #     spread = 0.05
    
    wing_size = 3
    
    put_K = np.linspace(s*(1-spread),s*0.995,wing_size)
    
    call_K = np.linspace(s*1.005,s*(1+spread),wing_size)
    
    
    # put_K = np.arange(
    #     s-(wing_size)*5, 
    #     s-5+1, 
    #     5
    #     )
    
    # call_K = np.arange(
    #     s+5, 
    #     s+(wing_size)*5+1, 
    #     5
    #     )
    
    
    
    K = np.unique(np.array([put_K,call_K]).flatten())
    
    T = [
        30,60,95,
        186,368
        ]
    
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
        calibration_dataset['days_to_maturity'],
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
        
    #     # T = np.arange(30,360,28)
        
    #     T = [186,368]
        
    #     K = np.linspace(s*0.95,s*1.05,1400)
        
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
        
    #     vanilla_features['kappa'] = heston_parameters['kappa']
    #     vanilla_features['theta']  = heston_parameters['theta']
    #     vanilla_features['eta'] = heston_parameters['eta']
    #     vanilla_features['rho'] = heston_parameters['rho']
    #     vanilla_features['v0'] = heston_parameters['v0']
        
    #     vanilla_features['volatility'] = ms.derman_volatilities(
    #         s, 
    #         vanilla_features['strike_price'],
    #         vanilla_features['days_to_maturity'],
    #         vanilla_features['days_to_maturity'].map(derman.derman_coefs), 
    #         vanilla_features['days_to_maturity'].map(atm_volvec)
    #         )
        
    #     vanilla_features['calculation_date'] = dtdate

    #     vanilla_features['expiration_date'] = expiration_dates
        
    #     vanilla_features['numpy_black_scholes'] = ms.vector_black_scholes(
    #             vanilla_features['spot_price'],
    #             vanilla_features['strike_price'],
    #             r,
    #             vanilla_features['days_to_maturity'],
    #             vanilla_features['volatility'],
    #             vanilla_features['w']
                
    #         )
        
    #     vanilla_features['ql_black_scholes'] = ms.vector_qlbs(
    #             vanilla_features['spot_price'],
    #             vanilla_features['strike_price'],
    #             r,g,
    #             vanilla_features['volatility'],
    #             vanilla_features['w'],
    #             calculation_date,
    #             vanilla_features['expiration_date']
    #         )
        
    #     vanilla_features['heston_price']  = ms.vector_heston_price(
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
        
    #     dt_expiration_dates = []
    #     for date in expiration_dates:
    #         dt_expiration_dates.append(
    #             datetime(date.year(),date.month(),date.dayOfMonth())
    #             )
        
    #     vanilla_features['expiration_date'] = dt_expiration_dates
    #     print(vanilla_features)
    #     file_time = datetime.fromtimestamp(time.time())
    #     file_tag = file_time.strftime("%Y-%m-%d %H%M%S")
    #     file_name = dtdate.strftime("%Y-%m-%d") \
    #         + f" spot{int(s)} " + file_tag + r".csv"
    #     vanilla_features.to_csv(os.path.join(vanilla_csv_dir,file_name))
    #     print(f"\n{vanilla_features.describe()}\n{print_date}")

        
    #     # """
        
    #     # # ############
    #     # # # BARRIERS #
    #     # # ############
    
    #     # # up_barriers = np.linspace(s*1.01,s*1.19,50)
    #     # # down_barriers = np.linspace(s*0.81,s*0.99,50)
        
    #     # # down_features = generate_barrier_features(
    #     # #     s,K,T,down_barriers,'Down', ['Out','In'], ['call','put']
    #     # #     )
        
    #     # # up_features = generate_barrier_features(
    #     # #     s,K,T,up_barriers,'Up', ['Out','In'], ['call','put']
    #     # #     )
        
    #     # # features = pd.concat([down_features,up_features],ignore_index=True)
    #     # # features['barrier_type_name'] = features['updown'] + features['outin']
        
    #     # # barrier_options = generate_barrier_options(
    #     # #     features,calculation_date,heston_parameters, g, r'hist_outputs')
        
    #     # # historical_contracts = pd.concat(
    #     # #     [historical_contracts, barrier_options],ignore_index=True)
     
    #     # """
        
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
        
plt.figure()
plt.plot(historical_calibration_errors,color = 'black')
plt.title("Hisotorical calibration error")
plt.ylabel("absolute relative error") 
print(f"\nhistorical average absolute relative calibration error: "
      f"{round(np.average(np.abs(calibration_error)),2)}%") 
        
            
            
            
            
            