# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:04:18 2024

@author: boomelage
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,'train_data'))
from settings import model_settings
ms = model_settings()

from train_generation_barriers import generate_barrier_options
# example use

from routine_calibration_testing import heston_parameters
title = 'barrier options'
# T = ms.T
T = [10,30,90]

n_strikes = 3
down_k_spread = 0.05
up_k_spread = 0.05

n_barriers = 3
barrier_spread = 0.0010                   
n_barrier_spreads = 5

g=0.001

training_data = generate_barrier_options(
        n_strikes, down_k_spread, up_k_spread,
        n_barriers, barrier_spread, n_barrier_spreads,
        ms.calculation_date, T, ms.s, g, heston_parameters, file_path='barriers'
            )

training_data