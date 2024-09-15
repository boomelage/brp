#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:23:22 2024

"""

import pandas as pd
import numpy as np

from data_query import dirdatacsv
csvs = dirdatacsv()
rawtsname = [file for file in csvs if 'raw_ts' in file][0]
raw_ts = pd.read_csv(rawtsname).drop_duplicates()
raw_ts = raw_ts.rename(
    columns={raw_ts.columns[0]: 'Strike'}).set_index('Strike')
raw_ts.columns = raw_ts.columns.astype(int)
raw_ts = raw_ts.replace(0,np.nan)
raw_ts = raw_ts/100

raw_K = np.sort(raw_ts.index)

s = raw_K[int(len(raw_K)/2)]

s