# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:48:31 2024

@author: boomelage
"""

from callsputs import option_data, ivol_table, ivols, implied_vols_matrix,\
n_maturities, n_strikes, dataset
from pricing import BS_price_vanillas, heston_price_vanillas
import numpy as np
import pandas as pd


dataset

bs = BS_price_vanillas(dataset)

bs