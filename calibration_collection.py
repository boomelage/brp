# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:20:53 2024

"""

from calibration import heston_parameters_alt
from routine_calibration import heston_parameters

print('\n'*5)
print("parameters calibrated with less than 1% pricing error:")
print(f"\n{heston_parameters}")
print(f"\n{heston_parameters_alt}")






