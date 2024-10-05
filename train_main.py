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
sys.path.append(os.path.join(
    current_dir,
    'historical_data',
    'historical_generation'))
pd.set_option("display.max_columns",None)
train_start = time.time()
train_start_datetime = datetime.fromtimestamp(train_start)
train_start_tag = train_start_datetime.strftime('%c')

print("\n"+"#"*18+"\n# training start #\n"+
      "#"*18+"\n"+f"\n{train_start_tag}\n")

"""
# =============================================================================
                                importing data
"""

from HDF_collection import contracts

dataset = contracts.copy()
mlop = mlop(user_dataset = dataset)

"""
# =============================================================================
                            preprocessing data
"""
train_data = dataset[
    (
      (dataset['calculation_date']>=datetime(2007,10,1))&
      (dataset['calculation_date']<=datetime(2009,4,1))
      )]
test_data = dataset[
    (
      (dataset['calculation_date']>=datetime(2009,4,2))&
      (dataset['calculation_date']<=datetime(2012,12,31))
      )]

train_X, train_y, test_X, test_y = mlop.split_data_manually(
    train_data, test_data)

# train_data, train_X, train_y, \
#     test_data, test_X, test_y = mlop.split_user_data()

preprocessor = mlop.preprocess()

print(f"\ntrain data:\n{int(train_data.describe())}\n")

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

insample_results, outofsample_results, errors =  mlop.test_prediction_accuracy(
        model_fit,
        test_data,
        train_data
        )

"""
# =============================================================================
"""

S = np.sort(contracts['spot_price'].unique())
K = np.sort(contracts['strike_price'].unique())
T = np.sort(contracts['days_to_maturity'].unique())
W = np.sort(contracts['w'].unique())
n_calls = contracts[contracts['w']=='call'].shape[0]
n_puts = contracts[contracts['w']=='put'].shape[0]

train_end_tag = str(datetime.fromtimestamp(
    train_end).strftime("%Y_%m_%d %H%M%S"))
file_tag = str(model_name + f" ntrain{train_data.shape[0]} " + train_end_tag)

os.chdir(current_dir)
os.mkdir(file_tag)
file_dir = os.path.join(current_dir,file_tag,file_tag)

joblib.dump(model_fit,str(f"{file_dir}.pkl"))

pd.set_option("display.max_columns",None)
with open(f'{file_dir}.txt', 'w') as file:
    file.write(f"\n{contracts}")
    file.write(f"\n{contracts.describe()}\n")
    file.write(f"\nspot(s):\n{S}")
    file.write(f"\n\nstrikes:\n{K}\n")
    file.write(f"\nmaturities:\n{T}\n")
    file.write(f"\ntypes:\n{W}\n")
    try:
        file.write(f"\n{contracts['barrier_type_name'].unique()}")
    except Exception:
        pass
    file.write("")
    file.write(
        f"\n\nmoneyness:\n{np.sort(contracts['moneyness'].unique())}\n")
    file.write(f"\nnumber of calls, puts:\n{n_calls},{n_puts}\n")
    file.write(f"\ntotal prices:\n{contracts.shape[0]}\n")
    for spec in specs:
        file.write(f"{spec}\n")
    file.write(
        f"\ntrain data:\n{train_data.describe()}\n\ntest data: \n"
        f"{test_data.describe()}")
    file.write(
        f"\nin sample results:"
        f"\n     RMSE: {errors['insample_RMSE']}"
        f"\n     MAE: {errors['insammple_MAE']}"
        f"\nout of sample results:"
        f"\n     RMSE: {errors['outofsample_RMSE']}"
        f"\n     MAE: {errors['outofsample_MAE']}\n"
        )
pd.reset_option("display.max_columns")
