
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:27:39 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)

import pandas as pd
from data_query import dirdatacsv
csvs = dirdatacsv()

dcname = [file for file in csvs if 'derman_coefs' in file][0]
derman_coefs = pd.read_csv(dcname)
derman_coefs = derman_coefs.set_index('coef')
derman_coefs.columns = derman_coefs.columns.astype(int)

dtsname = [file for file in csvs if 'derman_ts' in file][0]
derman_ts = pd.read_csv(dtsname).drop_duplicates()
derman_ts = derman_ts.rename(
    columns={derman_ts.columns[0]:'Strike'}).set_index('Strike')
derman_ts.columns = derman_ts.columns.astype(int)

stsname = [file for file in csvs if 'spread_ts' in file][0]
spread_ts = pd.read_csv(stsname).drop_duplicates()
spread_ts = spread_ts.rename(
    columns={spread_ts.columns[0]:'Strike'}).set_index('Strike')
spread_ts.columns = spread_ts.columns.astype(int)

rawtsname = [file for file in csvs if 'raw_ts' in file][0]
raw_ts = pd.read_csv(rawtsname).drop_duplicates()
raw_ts = raw_ts.rename(
    columns={raw_ts.columns[0]:'Strike'}).set_index('Strike')
raw_ts.columns = raw_ts.columns.astype(int)
imported_ts = {
    'derman_coefs': derman_coefs,
    'derman_ts': derman_ts,
    'spread_ts': spread_ts,
    'raw_ts': raw_ts,
    }
print('\nderman_coefs, derman_ts, spread_ts, raw_ts\n')


derman_coefs = imported_ts['derman_coefs']
derman_ts = imported_ts['derman_ts']
spread_ts = imported_ts['spread_ts']
raw_ts = imported_ts['raw_ts']


contdetname = [file for file in csvs if 'contract_details' in file][0]
contract_details = pd.read_csv(contdetname).drop_duplicates()
contract_details = contract_details.drop(
    columns = contract_details.columns[0])
print('\ncontract_details\n')


print('\nfiles imported\n')




