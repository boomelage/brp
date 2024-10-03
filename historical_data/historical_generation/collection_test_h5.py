#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:38:02 2024

@author: doomd
"""

import pandas as pd


with pd.HDFStore('SPXvanillas.h5') as store:
    call_keys = [key for key in store.keys() if key.startswith('/call/')]
    all_call_dataframes = [store[key] for key in call_keys]
    calls = pd.concat(all_call_dataframes, ignore_index=True)
    
calls