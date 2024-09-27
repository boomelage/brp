# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 00:19:20 2024

@author: boomelage
"""
from settings import model_settings
import QuantLib as ql
ms = model_settings()
bs = ms.ql_black_scholes(
        1416.59, 1133.272, 0.04, 0.018125,
        0.10538299999999999,'call',
        ql.Date(1,1,2007), 
        ql.Date(2,2,2007)
        )

bs