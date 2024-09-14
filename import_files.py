
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:27:39 2024

"""
import matplotlib.pyplot as plt
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

import pandas as pd
from data_query import dirdatacsv
csvs = dirdatacsv()

ticker = 'SPX'
lower_strike = 5620 # None
upper_strike = 5640 # None
lower_maturity = 31
upper_maturity = 380 #None

dcname = [file for file in csvs if 'derman_coefs' in file][0]
derman_coefs = pd.read_csv(dcname)
derman_coefs = derman_coefs.set_index('coef')
derman_coefs.columns = derman_coefs.columns.astype(int)

dtsname = [file for file in csvs if 'derman_ts' in file][0]
derman_ts = pd.read_csv(dtsname).drop_duplicates()
derman_ts = derman_ts.rename(
    columns={derman_ts.columns[0]:'Strike'}).set_index('Strike')
derman_ts.columns = derman_ts.columns.astype(int)
derman_ts = derman_ts.loc[
          lower_strike:upper_strike,lower_maturity:upper_maturity]

stsname = [file for file in csvs if 'spread_ts' in file][0]
spread_ts = pd.read_csv(stsname).drop_duplicates()
spread_ts = spread_ts.rename(
    columns={spread_ts.columns[0]:'Strike'}).set_index('Strike')
spread_ts.columns = spread_ts.columns.astype(int)
spread_ts = spread_ts.loc[
          lower_strike:upper_strike,lower_maturity:upper_maturity]

rawtsname = [file for file in csvs if 'raw_ts' in file][0]
raw_ts = pd.read_csv(rawtsname).drop_duplicates()
raw_ts = raw_ts.rename(
    columns={raw_ts.columns[0]:'Strike'}).set_index('Strike')
raw_ts.columns = raw_ts.columns.astype(int)
raw_ts = raw_ts.loc[
          lower_strike:upper_strike,lower_maturity:upper_maturity]


print('\nderman_coefs, derman_ts, spread_ts, raw_ts')
imported_ts = {
    'derman_coefs': derman_coefs,
    'derman_ts': derman_ts,
    'spread_ts': spread_ts,
    'raw_ts': raw_ts,
    }
derman_coefs = imported_ts['derman_coefs']
derman_ts = imported_ts['derman_ts']
spread_ts = imported_ts['spread_ts']
raw_ts = imported_ts['raw_ts']


"""
                       the above files need to be present to generate the below
"""

contdetname = [file for file in csvs if 'contract_details' in file][0]
contract_details = pd.read_csv(contdetname).drop_duplicates()
contract_details = contract_details.drop(
    columns = contract_details.columns[0])


from surface_plotting import plot_volatility_surface, plot_term_structure
from settings import model_settings
ms = model_settings()
settings = ms.import_model_settings()
dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']



T = derman_ts.columns
K = derman_ts.index
expiration_dates = ms.compute_ql_maturity_dates(T)
implied_vols_matrix = ms.make_implied_vols_matrix(K, T, derman_ts)
black_var_surface = ms.make_black_var_surface(expiration_dates, K, implied_vols_matrix)
fig = plot_volatility_surface(black_var_surface, K, T)
for t in T:
    fig = plot_term_structure(K, t, spread_ts, derman_ts)
    plt.cla()
    plt.clf()
print("\nvolatility surface generated")
print("\ncontract_details\n\nfiles imported")
print(f"\nvolatility surface:\n{derman_ts}")

spread_ts
