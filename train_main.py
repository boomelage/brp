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

import train_contracts

title = 'Prediction errors'

dataset = train_contracts.training_data.copy()
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


# model_fit, runtime = mlop.run_rf(preprocessor,train_X,train_y)


"""
lasso regression
"""


# model_fit, runtime = mlop.run_lm(train_X,train_y)



""""""
estimation_end_time = time.time()

"""
# =============================================================================
                                model testing
"""

stats = mlop.test_model(
    test_data, test_X, test_y, model_fit)

predictive_performance_plot = mlop.plot_model_performance(
    stats,runtime,title)


"""
# =============================================================================
"""

estimation_end_tag = str(datetime.fromtimestamp(
    estimation_end_time).strftime(
        "%Y-%m-%d %H%M%S")
        )
file_name = str(
    model_name + " " + estimation_end_tag + f" ser{np.random.randint(1,999)}"
    )
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(file_name, exist_ok=True)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),file_name))
joblib.dump(model_fit,str(f"{file_name}.pkl"))
pd.set_option("display.max_columns",None)
with open(f"{file_name}.txt", "w") as f:
    f.write(f"\n{dataset}\n")
    f.write(f"\n{dataset.describe()}\n")
    f.write(
        f"\nspot(s):\n{train_contracts.S}\n\nstrikes:\n{train_contracts.K}\n\n")
    f.write(f"maturities:\n{train_contracts.T}\n\ntypes:\n{train_contracts.W}\n")
    f.write(f"\n{dataset['moneyness_tag'].unique()}\n")
    try:
        f.write(f"\n{dataset['barrier_type_name'].unique()}")
    except Exception:
        pass
    f.write(f"\nmoneyness:\n{np.sort(dataset['moneyness'].unique())}\n")
    f.write("\nnumber of calls, puts:")
    f.write("\n{train_contracts.n_calls},{train_contracts.n_puts}\n")
    f.write(f"\ninitial count:\n{train_contracts.initial_count}\n")
    f.write(f"\ntotal prices:\n{dataset.shape[0]}\n")
    f.write(f"\n{stats.describe()}\n")
    for spec in specs:
        f.write(f"\n{spec}")
pd.reset_option("display.max_columns")




