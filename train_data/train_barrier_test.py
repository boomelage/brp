# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:45:34 2024

@author: boomelage
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from settings import model_settings
from routine_calibration_testing import heston_parameters
ms = model_settings()
os.chdir(current_dir)
from train_generation_barriers import concat_barrier_features,\
    generate_barrier_options
    
T = [10,30,90]

n_strikes = 5
down_k_spread = -0.1
up_k_spread = 0.1

n_barriers = 5
barrier_spread = 0.005                 
n_barrier_spreads = 20

s=ms.s

g = 0.00

features = concat_barrier_features(
        s,T,g,heston_parameters,
        down_k_spread, up_k_spread, n_strikes,
        barrier_spread,n_barrier_spreads,n_barriers
        )

df = generate_barrier_options(
        features, ms.calculation_date, heston_parameters, 0.00, r'barriers'
        )


