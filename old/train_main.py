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
manual filtering
"""
dataset = dataset[dataset['outin']=='Out']

# leave a space or other separator in front
file_tag_addon = " OnlyOuts"

"""
random train/test split
"""
# train_data, train_X, train_y, \
#     test_data, test_X, test_y = mlop.split_user_data()
""""""

"""
manual train/test split
"""
unique_dates = dataset['calculation_date'].sort_values(
    ascending=True).unique().tolist()
filter_date = unique_dates[int(0.9*len(unique_dates))]

train_data = dataset[
    (
      # (dataset['calculation_date']>=datetime(2007,1,1))
      #  &
        (dataset['calculation_date']<=filter_date)
      )].copy()

test_data = dataset[
    (
      (dataset['calculation_date']>filter_date)
      # &
      # (dataset['calculation_date']<=datetime(2012,12,31))
      )].copy()

train_X, train_y, test_X, test_y = mlop.split_data_manually(dataset,
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
print(f"\ntotal prices:\n{train_data.shape[0]}")
print(f"\n{train_data.dtypes}")
pd.reset_option("display.max_columns")

test_train_ratio = test_data.describe(
    ).iloc[0,0]/train_data.describe().iloc[0,0]
print(f"\ntrain/test: {int(round((1-test_train_ratio)*100,0))}/"
      +str(int(round(test_train_ratio*100,0))))


"""
# =============================================================================
                              model selection                

single layer network
"""



# model_fit, runtime, specs = mlop.run_nnet(preprocessor, train_X, train_y)



"""
deep neural network
"""



# model_fit, runtime, specs = mlop.run_dnn(preprocessor,train_X,train_y)



"""
random forest
"""



model_fit, runtime, specs = mlop.run_rf(preprocessor,train_X,train_y)



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
insample, outsample, errors = mlop.test_prediction_accuracy(
        model_fit,
        test_data,
        train_data
        )

"""
# =============================================================================
"""

