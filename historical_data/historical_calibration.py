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
import Derman as derman
from routine_calibration_global import calibrate_heston
from routine_calibration_testing import test_heston_calibration
from model_settings import ms
os.chdir(current_dir)

from historical_collection import historical_data
historical_data = historical_data.copy()


"""
# =============================================================================
                        historical calibration routine
"""
param_names = ['theta','kappa','rho','eta','v0','relative_error']
historical_data.loc[:,param_names] = np.nan
historical_data
hist_calculation_datetimes = np.array(
    historical_data['date'],
    dtype=object
    )


historical_calibration_errors = pd.Series()
test_vols = pd.Series(
    np.zeros(len(hist_calculation_datetimes),dtype=float),
    index=hist_calculation_datetimes
    )
for row_i, row in historical_data.iterrows():
    s = row['spot_price']
    g = row['dividend_rate']
    r = 0.04
    calculation_datetime = row['date']
    print_date = calculation_datetime.strftime('%A %d %B %Y')
    calculation_date = ql.Date(
        calculation_datetime.day,
        calculation_datetime.month,
        calculation_datetime.year)
    
    """
    ###############
    # CALIBRATION #
    ###############
    """   
             
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
        ],dtype=float)
    
    atm_volvec = pd.Series(atm_volvec)
    atm_volvec.index = T

    if calculation_datetime == datetime(2008,11,20):
        spread = 0.30
    elif s < 900:
        spread = 0.25
    elif s < 1000:
        spread = 0.20
    elif s < 1100:
        spread = 0.15
    elif s < 1200:
        spread = 0.10
    else:
        spread = 0.05
    
    wing_size = 3
    
    put_K = np.linspace(s*(1-spread),s*0.995,wing_size)
    
    call_K = np.linspace(s*1.005,s*(1+spread),wing_size)
    
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
    
    calibration_dataset['risk_free_rate'] = r
    calibration_dataset['dividend_rate'] = g
    
    heston_parameters = calibrate_heston(
        calibration_dataset, s, r, g, calculation_date)
    
    time.sleep(0.01)
    
    heston_parameters = test_heston_calibration(
        calibration_dataset, heston_parameters, calculation_datetime, r, g)
    
    historical_data.loc[row_i,param_names] = heston_parameters[param_names]
    
    print(f"^ {print_date}")
    
    calibration_error = heston_parameters['relative_error']
    test_vols[calculation_datetime] = atm_volvec.iloc[0]
    historical_calibration_errors[calculation_datetime] = calibration_error
    
print(f"\nhistorical average absolute relative calibration error: "
      f"{round(np.average(np.abs(calibration_error*100)),4)}%")       

historical_data.to_csv(
    os.path.join(
        'historical_generation',
        '2007-2012 SPX calibrated.csv'
        )
    )
print("\nhistorical parameters saved!")



plt.figure()
plt.plot(
    historical_calibration_errors,
    color = 'black'
    )
plt.title("Hisotorical calibration error")
plt.ylabel("absolute relative error") 
plt.show()
plt.clf



