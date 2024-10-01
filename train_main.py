# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:13:18 2024

"""
import os
import sys
import time
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
from mlop import mlop
sys.path.append(os.path.join(current_dir,'train_data'))

train_start = time.time()
train_start_datetime = datetime.fromtimestamp(train_start)
train_start_tag = train_start_datetime.strftime('%c')
print(f"\n{train_start_tag}\n")

"""
# =============================================================================
                                importing data
"""

from train_vanillas import training_data
title = 'Prediction errors for vanilla options'

mlop = mlop(user_dataset = training_data)

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

model_fit, runtime = mlop.run_dnn(preprocessor,train_X,train_y)

"""
random forest
"""

# model_fit, runtime = mlop.run_rf(preprocessor,train_X,train_y)

"""
lasso regression
"""

# model_fit, runtime = mlop.run_lm(train_X,train_y)

"""
# =============================================================================
                                model testing
"""

stats = mlop.test_model(test_data, test_X, test_y, model_fit)

predictive_performance_plot = mlop.plot_model_performance(stats,runtime,title)

"""
# =============================================================================
"""

train_end = time.time()
train_time = train_end - train_start

print(f"\nruntime: {int(train_time)} seconds")

