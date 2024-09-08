#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
"""
Created on Sun Sep  8 03:09:26 2024

"""

from bloomberg_ivols import generate_from_market_data    

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm




# dataset, ivol_table, implied_vols_matrix, black_var_surface, strikes,\
#     maturities = generate_from_market_data(0.00, 0.00)

maxK = max(strikes)
minK = min(strikes)
theoretical_strikes = np.linspace(minK,maxK,int((maxK-minK)*3))
n_theostrikes = len(theoretical_strikes)
theoretical_maturities = np.arange(14/365, 2.01,7/365)
n_theomats = len(theoretical_maturities)

from generate_ivols import generate_ivol_table
theoretical_ivol_table = generate_ivol_table(
    n_theomats, n_theostrikes, max(max(ivol_table)))




