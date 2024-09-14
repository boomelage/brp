#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:04:00 2024

@author: doomd
"""
import time
from datetime import datetime

ticker = r"SPX"
from rountine_Derman import raw_ts, derman_ts, spread_ts, K, T, s, derman_coefs

file_time = time.time()
file_datetime = datetime.fromtimestamp(file_time)
# time_tag = file_datetime.strftime('%Y-%m-%d %H-%M-%S')
time_tag = '2024-09-13'

generic = f"{ticker} {time_tag}"
raw_ts.to_csv(f"{generic} raw_ts.csv")
spread_ts.to_csv(f"{generic} rspread_ts.csv")
derman_ts.to_csv(f"{generic} derman_ts.csv")
derman_coefs.to_csv(f"{generic} derman_coefs.csv")
