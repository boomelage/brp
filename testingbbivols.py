#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:00:04 2024

@author: doomd
"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
from datapwd import dirdata
import QuantLib as ql
import warnings
warnings.simplefilter(action='ignore')
data_files = dirdata()
import numpy as np


pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns

# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')

from bloomberg_ivols import bloomberg_ivols

dividend_rate = 0.00
risk_free_rate = 0.00

bivs = bloomberg_ivols(dividend_rate, data_files, risk_free_rate)

dataset, ivol_table, implied_vols_matrix, \
    black_var_surface, strikes, maturities = bivs.generate_from_market_data()