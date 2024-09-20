# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:12:00 2024

@author: boomelage
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append('term_structure')
sys.path.append('contract_details')
import time
from datetime import datetime
import pandas as pd
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
from train_generation import ml_data

# ml_data = ml_data[ml_data['heston_price']<0]

from settings import model_settings
ms = model_settings()

heston_pricer = ms.heston_price_one_vanilla
calendar = ms.calendar
day_count = ms.day_count
calculation_date = ms.calculation_date


for i, row in ml_data.iterrows():
    spot_price = row['spot_price']
    strike_price = row['strike_price']
    days_to_maturity = row['days_to_maturity']
    w = row['w']
    dividend_rate = row['dividend_rate']
    risk_free_rate = row['risk_free_rate']
    sigma = row['sigma']
    theta = row['theta']
    kappa = row['kappa']
    rho = row['rho']
    v0 = row['v0']
    volatility = row['volatility']
    
    h_price = ms.heston_price_one_vanilla(
        spot_price, strike_price, days_to_maturity, risk_free_rate, \
            dividend_rate, v0, kappa, theta, sigma, rho, w)
    file_write_time = time.time()
    file_write_dt = datetime.fromtimestamp(file_write_time)
    write_tag = file_write_dt.strftime("%Y-%m-%d %H-%M-%S")
    if h_price < 0:
        with open(f'{write_tag} test.txt', 'a') as file:
            output_str = f"\n\n{row}\nHeston Price: {h_price}"
            file.write(output_str)
            print(output_str)
    else:
        pass
