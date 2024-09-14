
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:27:39 2024

"""
from routine_collection import collect_directory_market_data
from settings import model_settings
from data_query import dirdatacsv
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

csvs = dirdatacsv()
ms = model_settings()
settings = ms.import_model_settings()
dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']
security_settings = settings['security_settings']
ticker = security_settings[0]
lower_strike = security_settings[1]
upper_strike = security_settings[2]
lower_maturity = security_settings[3]
upper_maturity = security_settings[4]
s = security_settings[5]


file_time = time.time()
file_datetime = datetime.fromtimestamp(file_time)
time_tag = file_datetime.strftime('%H-%M-%S')
date_tag = file_datetime.strftime('%Y-%m-%d')
date_tag = '2024-09-13'
generic = f"{ticker} {date_tag} {time_tag}"

"""
# =============================================================================
                                                                   import files
                                                                   
                                                                         raw_ts
"""

rawtsname = [file for file in csvs if 'raw_ts' in file][0]
raw_ts = pd.read_csv(rawtsname).drop_duplicates()
raw_ts = raw_ts.rename(
    columns={raw_ts.columns[0]: 'Strike'}).set_index('Strike')
raw_ts.columns = raw_ts.columns.astype(int)
raw_ts = raw_ts.loc[
    lower_strike:upper_strike,
    lower_maturity:upper_maturity]


"""
                                                               contract_details
"""

# contract_details_name = [
#     file for file in csvs if 'contract_details' in file][0]
# contract_details = pd.read_csv(contract_details_name).drop_duplicates()
# contract_details = contract_details.drop(
#     columns = contract_details.columns[0])

"""
                                                                         Derman
"""

# derman_ts_name = [file for file in csvs if 'derman_coefs' in file][0]
# derman_coefs = pd.read_csv(derman_ts_name).set_index('coef')

"""
                                                                       plotting
"""

# from surface_plotting import plot_volatility_surface, plot_term_structure
# T = derman_ts.columns
# K = derman_ts.index
# expiration_dates = ms.compute_ql_maturity_dates(T)
# implied_vols_matrix = ms.make_implied_vols_matrix(K, T, derman_ts)
# black_var_surface = ms.make_black_var_surface(expiration_dates, K, implied_vols_matrix)
# fig = plot_volatility_surface(black_var_surface, K, T)
# for t in T:
#     fig = plot_term_structure(K, t, spread_ts, derman_ts)
#     plt.cla()
#     plt.clf()


"""
# =============================================================================
                                                                     save files
                                                                   
                                                                         raw_ts
"""



# from routine_ivol_collection import raw_ts
# raw_ts.drop_duplicates().to_csv(f"{generic} raw_ts.csv")

"""
                                                                         Derman
"""

# from rountine_Derman import derman_coefs
# derman_coefs.drop_duplicates().to_csv(f"{generic} derman_coefs.csv")

"""
                                                               contract_details
"""
contract_details = collect_directory_market_data()
# contract_details.drop_duplicates().to_csv(f"{generic} contract_details.csv")
