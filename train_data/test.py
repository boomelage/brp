# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:04:18 2024

@author: boomelage
"""
import os
import sys
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,'train_data'))
from settings import model_settings
ms = model_settings()


T = [
      10,
      # 30,90
      ]

n_strikes = 5
down_k_spread = 0.05
up_k_spread = 0.05

n_barriers = 5
barrier_spread = 0.0010                   
n_barrier_spreads = 5

g = 0.001
s = ms.s

K = np.linspace(
    s*(1-down_k_spread),
    s*(1+up_k_spread),
    n_strikes
    )

calculation_date = ms.calculation_date

from train_generation_barriers import generate_barrier_options, concat_barrier_features
from routine_calibration_testing import heston_parameters

features = concat_barrier_features(
        s,K,T,g,heston_parameters,
        barrier_spread,n_barrier_spreads,n_barriers)

training_data = generate_barrier_options(
    features, calculation_date, heston_parameters, g)