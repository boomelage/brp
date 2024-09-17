# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 08:36:30 2024

@author: boomelage
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
from import_files import raw_ts

complete_market_ts = raw_ts.dropna(how= 'all',axis = 0)
complete_market_ts = raw_ts.dropna(how= 'all',axis = 1)


K = complete_market_ts.iloc[:,1].dropna().index

trimmed_market_ts = complete_market_ts.loc[K,:]


trimmed_market_ts = trimmed_market_ts.dropna(how='any',axis=1)


