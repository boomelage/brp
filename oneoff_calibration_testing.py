# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:29:31 2024

example calibration

"""


from routine_calibration_generation import calibration_dataset
from routine_calibration_global import calibrate_heston
from Derman import derman_s
s = derman_s
calculation_date = ms.calculation_date
ql.Settings.instance().evaluationDate = calculation_date

g = 0.001
r = 0.04

heston_parameters = calibrate_heston(
    calibration_dataset, s, r, g, calculation_date)

test_features = calibration_dataset.copy()

heston_parameters = test_heston_calibration(
        test_features, heston_parameters,calculation_date,r,g)
