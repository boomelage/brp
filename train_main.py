# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:13:18 2024

"""
import os
import sys
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
from mlop import mlop
sys.path.append(os.path.join(current_dir,'train_data'))

train_start = time.time()
train_start_datetime = datetime.fromtimestamp(train_start)
train_start_tag = train_start_datetime.strftime('%c')

print("\n"+"#"*18+"\n# training start #\n"+
      "#"*18+"\n"+f"\n{train_start_tag}\n")

"""
# =============================================================================
                                importing data
"""

from train_contracts import training_data

title = 'Prediction errors'

dataset = training_data.copy()
mlop = mlop(user_dataset = dataset)

"""
# =============================================================================
                            preprocessing data
"""

train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.split_user_data()

preprocessor = mlop.preprocess()

print(f"\ntrain data count:\n{int(train_data.shape[0])}")

"""
# =============================================================================
                              model selection                

single layer network
"""


# model_fit, runtime = mlop.run_nnet(preprocessor, train_X, train_y)


"""
deep neural network
"""


model_fit, runtime, specs = mlop.run_dnn(preprocessor,train_X,train_y)
model_name = r'deep_neural_network'


"""
random forest
"""

# model_fit, runtime, specs = mlop.run_rf(preprocessor,train_X,train_y)
# model_name = r'random_forest'

"""
lasso regression
"""


# model_fit, runtime = mlop.run_lm(train_X,train_y)



""""""
train_end = time.time()
train_runtime = train_end-train_start
"""
# =============================================================================
                                model testing
"""

stats = mlop.test_model(
    test_data, test_X, test_y, model_fit)


"""
# =============================================================================
"""

sqdifference = (stats['prediciton']-np.mean(stats['target']))**2
absdifference = np.abs(stats['prediciton']-np.mean(stats['target']))
RSME = np.sqrt(np.average(sqdifference))
MAE = np.average(absdifference)
print(f"\nRSME: {RSME}\nMAE: {MAE}")
print(f"\ntrain runtime: {round(train_runtime,3)} seconds")
