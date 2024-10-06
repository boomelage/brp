# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:10:20 2024

@author: boomelage
"""
import modin.pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import time
from historical_barrier_feature_generation import \
    barrier_puts_by_date, barrier_calls_by_date,barrier_puts,barrier_calls
    
    
    
# with pd.HDFStore('SPX barriers review.h5') as store:
    
for date in barrier_puts['calculation_date'].drop_duplicates():
    print(date)
    