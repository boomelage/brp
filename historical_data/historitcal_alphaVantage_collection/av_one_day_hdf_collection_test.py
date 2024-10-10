# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:02:17 2024

@author: boomelage
"""

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
import pandas as pd
from tqdm import tqdm


store = pd.HDFStore(r'alphaVantage vanillas.h5')
print(f"\n\n")
print(store.keys())
print(f"\n\n")
key = store.keys()[2]

print(store[key].columns)
print(f"\n\n")

store.close()