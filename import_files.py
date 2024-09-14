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


derman_coefs = pd.read_csv(csvs[0])
derman_coefs = derman_coefs.set_index('coef')
derman_coefs.columns = derman_coefs.columns.astype(int)


derman_ts = pd.read_csv(csvs[2])
derman_ts = derman_ts.rename(
    columns={derman_ts.columns[0]:'Strike'}).set_index('Strike')
derman_ts.columns = derman_ts.columns.astype(int)


spread_ts = pd.read_csv(csvs[4])
spread_ts = spread_ts.rename(
    columns={spread_ts.columns[0]:'Strike'}).set_index('Strike')
spread_ts.columns = spread_ts.columns.astype(int)


raw_ts = pd.read_csv(csvs[3])
raw_ts = raw_ts.rename(
    columns={raw_ts.columns[0]:'Strike'}).set_index('Strike')
raw_ts.columns = raw_ts.columns.astype(int)


contract_details = pd.read_csv(csvs[1])
contract_details = contract_details.drop(
    columns = contract_details.columns[0])
