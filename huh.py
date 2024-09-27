# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 00:19:20 2024

"""
from settings import model_settings
import QuantLib as ql
ms = model_settings()
bs = ms.ql_black_scholes(
        1416.59, 1133.272, 0.04, 0.018125,
        0.10538299999999999, 'call',
        ql.Date(1,1,2007), 
        ql.Date(2,2,2007)
        )



# bs = ms.ql_black_scholes(
#         100, 90, 0.04, 0,
#         0.20, 
#         'put',
#         ql.Date(1,1,2023), 
#         ql.Date(1,1,2024)
#         )
t = ql.Date(2,2,2007) - ql.Date(1,1,2007)
my_bs = ms.black_scholes_price(
    1416.59, 1133.272, t, 0.04, 0.10538299999999999, 'call')

print(f"\n{bs}\n{my_bs}")