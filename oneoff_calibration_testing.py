# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:29:31 2024

example calibration

"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import QuantLib as ql
from routine_calibration_global import calibrate_heston
from routine_calibration_testing import test_heston_calibration


from routine_calibration_generation import calibration_dataset, calculation_date
ql.Settings.instance().evaluationDate = calculation_date
s = calibration_dataset['spot_price'].unique()[0]
g = 0.001
r = 0.04

heston_parameters = calibrate_heston(
    calibration_dataset, s, r, g, calculation_date)

heston_parameters = test_heston_calibration(
        calibration_dataset, heston_parameters, calculation_date,r,g)



