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

random train/test split
"""
# train_data, train_X, train_y, \
#     test_data, test_X, test_y = mlop.split_user_data()
""""""


"""
manual train/test split
"""
unique_dates = dataset['calculation_date'].unique().tolist()
date75 = unique_dates[int(0.75*len(unique_dates))]

train_data = dataset[
    (
      # (dataset['calculation_date']>=datetime(2007,1,1))
      #  &
        (dataset['calculation_date']<=date75)
      )].copy()

test_data = dataset[
    (
      (dataset['calculation_date']>date75)
      # &
      # (dataset['calculation_date']<=datetime(2012,12,31))
      )].copy()

train_X, train_y, test_X, test_y = mlop.split_data_manually(
    train_data, test_data)

preprocessor = mlop.preprocess()

pd.set_option("display.max_columns",None)
print("#"*17+"\n# training data #\n"+"#"*17+
      f"\n{train_data.describe()}\n")


S = np.sort(train_data['spot_price'].unique())
K = np.sort(train_data['strike_price'].unique())
T = np.sort(train_data['days_to_maturity'].unique())
W = np.sort(train_data['w'].unique())
n_calls = train_data[train_data['w']=='call'].shape[0]
n_puts = train_data[train_data['w']=='put'].shape[0]

print(f"\nspot(s):\n{S}\n\nstrikes:\n{K}\n\nmaturities:\n{T}\n\ntypes:\n{W}")
try:
    print(f"\n{train_data['barrier_type_name'].unique()}")
except Exception:
    pass
print(f"\nnumber of calls, puts:\n{n_calls},{n_puts}")
print(f"\ntotal prices:\n{train_data.shape[0]}\n")
print(f"\n{train_data.dtypes}\n")
pd.reset_option("display.max_columns")



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

pd.set_option("display.max_columns",None)
print()
print("#"*13+"\n# test data #\n"+"#"*13+
      f"\n{test_data.describe()}\n")
insample_results, outofsample_results, errors =  mlop.test_prediction_accuracy(
        model_fit,
        test_data,
        train_data
        )

"""
# =============================================================================
"""

S = np.sort(train_data['spot_price'].unique())
K = np.sort(train_data['strike_price'].unique())
T = np.sort(train_data['days_to_maturity'].unique())
W = np.sort(train_data['w'].unique())
n_calls = train_data[train_data['w']=='call'].shape[0]
n_puts = train_data[train_data['w']=='put'].shape[0]

train_end_tag = str(datetime.fromtimestamp(
    train_end).strftime("%Y_%m_%d %H%M%S"))
file_tag = str(train_end_tag + " " + model_name + f" ntrain{train_data.shape[0]}")

os.chdir(current_dir)
os.mkdir(file_tag)
file_dir = os.path.join(current_dir,file_tag,file_tag)

joblib.dump(model_fit,str(f"{file_dir}.pkl"))

pd.set_option("display.max_columns",None)
with open(f'{file_dir}.txt', 'w') as file:
    file.write(f"\n{train_data}")
    file.write(f"\n{train_data.describe()}\n")
    file.write(f"\nspot(s):\n{S}")
    file.write(f"\n\nstrikes:\n{K}\n")
    file.write(f"\nmaturities:\n{T}\n")
    file.write(f"\ntypes:\n{W}\n")
    try:
        file.write(f"\n{train_data['barrier_type_name'].unique()}")
    except Exception:
        pass
    file.write("")
    file.write(f"\nnumber of calls, puts:\n{n_calls},{n_puts}\n")
    file.write(f"\ntotal prices:\n{train_data.shape[0]}\n")
    for spec in specs:
        file.write(f"{spec}\n")
    file.write("#"*17+"\n# training data #\n"+"#"*17+
          f"\n{train_data.describe()}\n")
    file.write("#"*13+"\n# test data #\n"+"#"*13+
          f"\n{test_data.describe()}\n")
    file.write(
        f"\nin sample results:"
        f"\n     RMSE: {errors['insample_RMSE']}"
        f"\n     MAE: {errors['insample_MAE']}\n"
        f"\nout of sample results:"
        f"\n     RMSE: {errors['outofsample_RMSE']}"
        f"\n     MAE: {errors['outofsample_MAE']}\n"
        )
    file.write("features:\n")
    for feature in mlop.feature_set:
        file.write(feature)
        file.write(f"\ntarget: {mlop.target_name}")
pd.reset_option("display.max_columns")



