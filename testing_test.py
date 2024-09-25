# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:08:33 2024

@author: boomelage
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')
from settings import model_settings
ms = model_settings()
s = ms.s
from routine_calibration_generation import calibration_dataset
from routine_calibration_testing import test_heston_calibration

error_df = test_heston_calibration(calibration_dataset,s)