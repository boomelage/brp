# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:46:00 2024

@author: boomelage
"""

import os
import pandas as pd
import time
import joblib
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
from train_main import test_X, test_data, test_y, model_fit

training_results = test_X.copy()
training_results['moneyness'] = test_data.loc[test_X.index,'moneyness']
training_results['target'] = test_y
training_results['prediciton'] = model_fit.predict(test_X)
training_results['abs_relative_error'] = abs(
    training_results['prediciton']/training_results['target']-1)

descriptive_stats = training_results['abs_relative_error'].describe()
test_count = int(descriptive_stats['count'])
descriptive_stats = descriptive_stats[1:]
pd.set_option('display.float_format', '{:.10f}'.format)
print(
    f"\nresults:\n--------\ntest data count: {test_count}"
    f"\n{descriptive_stats}\n"
    )
pd.reset_option('display.float_format')

dnn_end = time.time()
dnn_end_tag = str(datetime.fromtimestamp(
    dnn_end).strftime("%Y-%m-%d %H%M%S"))
joblib.dump(model_fit, str(f"dnn_model {dnn_end_tag}.pkl"))