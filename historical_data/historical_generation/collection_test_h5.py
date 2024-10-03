#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

@author: doomd
"""
import pandas as pd
with pd.HDFStore('SPXvanillas.h5') as store:
    all_call_keys = [key for key in store.keys() if key.startswith('/call/')]
    calls = pd.concat([store[key] for key in all_call_keys], ignore_index=True)
    
with pd.HDFStore('SPXvanillas.h5') as store:
    all_call_keys = [key for key in store.keys() if key.startswith('/put/')]
    puts = pd.concat([store[key] for key in all_call_keys], ignore_index=True)
contracts = pd.concat([calls,puts]).drop_duplicates().reset_index(drop=True)

contracts