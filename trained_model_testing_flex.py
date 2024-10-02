# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:46:00 2024

@author: boomelage
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime



current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
from settings import model_settings
ms = model_settings()
dnn_vanilla_price = joblib.load(
    os.path.join(current_dir,'dnn_model 2024-10-01 212117.pkl')
    )


test_start_time = time.time()
test_start_tag = datetime.fromtimestamp(
    test_start_time).strftime("%c")

print(test_start_tag)
print(dnn_vanilla_price)

s = 1391
S = [s]
K = np.linspace(s*0.9,s*1.1,int(1e3)).astype(int)
T = np.arange(31,180,7)
W = ['put']

test_df = pd.DataFrame(
    product(
        S, 
        K,
        T,
        W
        ),
    columns = [
        'spot_price','strike_price','days_to_maturity','w'
        ]
    )
test_df['moneyness'] = ms.vmoneyness(
    test_df['spot_price'], test_df['strike_price'], test_df['w']
    )

test_df['predicted'] = dnn_vanilla_price.predict(
    test_df[
        ['spot_price',
         'strike_price',
         'days_to_maturity',
         'moneyness']
        ]
    )

print(f"\n{test_df}")


test_end_time = time.time()
test_end_tag = datetime.fromtimestamp(
    test_end_time).strftime("%c")

test_runtime = test_end_time-test_start_time

print(
      f"\n{test_end_tag}\ntest runtime: {round(test_runtime,3)} seconds"
      )