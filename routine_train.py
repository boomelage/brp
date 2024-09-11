#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:39:41 2024

This is the principal file with which the model is estimated

"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,\
    MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, \
        SplineTransformer, PolynomialFeatures, KernelCenterer, \
            QuantileTransformer
import time
import textwrap
import os
from datetime import datetime
from mlop import mlop
import matplotlib.pyplot as plt

# =============================================================================
                                                             # General Settings
                                                          
target_name = 'observed_price'
security_tag = 'vanilla options'
feature_set = [
    
    'spot_price', 
    'dividend_rate', 
    'risk_free_rate',
    'days_to_maturity', 
    'strike_price'
    
    ]


model_scaler = [
                # RobustScaler(),
                QuantileTransformer(),
                # MaxAbsScaler(),
                # MinMaxScaler(),
                # Normalizer(),
                # PowerTransformer(),
                # SplineTransformer(),
                # PolynomialFeatures(),
                # KernelCenterer(),
                StandardScaler(),
                ""
                ]

transformers=[
    ("transformation_1",model_scaler[0],feature_set),
    ("transformation_2", model_scaler[1],feature_set)
    ]     

random_state = 42
test_size = 0.01

                                                      # Neural Network Settings
max_iter = 10000
activation_function = [        
    # 'identity', da
    # 'logistic',
    # 'tanh',
    'relu',
    ]
hidden_layer_sizes=(100, 100, 100)
solver= [
            # "lbfgs",
            "sgd",
            # "adam"
        ]
alpha = 0.0001
learning_rate = 'adaptive'

                                                       # Random Forest Settings
rf_n_estimators = 50
rf_min_samples_leaf = 2000

# =============================================================================
                                                                 # loading data

from routine_collection import collect_market_data_and_price
excluded_file = r'SPXts.xlsx'
excluded_file_format = f"\nTerm sturcutre: {excluded_file}"
print(excluded_file)
dataset = collect_market_data_and_price(excluded_file)
n_prices = f"\nestimated with {str(len(dataset))} "\
    f"option prices collected from the market"
print(n_prices)

# from routine_generation import dataset
# n_prices = f"estimated with {len(dataset)} synthesized option prices"
# print(f"\n{str(n_prices)}")

# =============================================================================
start_time = time.time()
start_tag = datetime.fromtimestamp(time.time())
start_tag_format = f"\n{str(start_tag.strftime('%c'))}"
start_tag = start_tag.strftime('%d%m%Y-%H%M%S')
print(start_tag_format)
model_scaler1 = model_scaler[0]
model_scaler2 = model_scaler[1]
scaler1name = str(f"{str(model_scaler[0])[:-2]}")
scaler2name = str(f"{str(model_scaler[1])[:-2]}")                                                            
dataset = dataset.dropna()
activation_function = activation_function[0]
solver = solver[0]
mlop = mlop(
    random_state = random_state,
    max_iter = max_iter,
    test_size = test_size,
    hidden_layer_sizes = hidden_layer_sizes,
    solver = solver,
    alpha = alpha,
    learning_rate = learning_rate,
    rf_n_estimators = rf_n_estimators,
    rf_min_samples_leaf = rf_min_samples_leaf,
    target_name = target_name,
    security_tag = security_tag,
    feature_set = feature_set,
    user_dataset = dataset,
    transformers = transformers,
    model_scaler1 = model_scaler1,
    model_scaler2 = model_scaler2
)

feature_str_list = '\n'.join(feature_set)
model_settings = (
    f"\nSelected Parameters:\nFeatures:\n{feature_str_list}\n\nTarget: "
    f"{target_name}\n\nSecurity: {security_tag}\n"
    )
print(model_settings)

# =============================================================================
                                                           # Preprocessing Data                                                 
train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.split_user_data()
    
    
preprocessor = mlop.preprocess()
# =============================================================================
                                                              # Model Selection
activation_function_tag = f'\nActivation function: {activation_function}'
print(activation_function_tag)
max_iter_tag = f'\nMaximum iterations: {max_iter}'
print(max_iter_tag)      
model_name = f"\nSingle Layer Network ({scaler1name}{scaler2name})"                           
model_fit, model_runtime = mlop.run_nnet(
    preprocessor, train_X, train_y, model_name, solver, hidden_layer_sizes, 
    activation_function, max_iter, random_state)

# model_name = f"{hidden_layer_sizes} Deep Neural Network "\
# f"({activation_function}) ({scaler1name}{scaler2name}) ({solver})"
# ml_settings = (
#     f"\n{datetime.fromtimestamp(time.time())}\n\nSelected Parameters:\n"
#     f"\nScaler: {scaler1name}{scaler2name}"
#     f"\nActivation function: {activation_function}"
#     f"\nMaximum iterations: {max_iter}"
#     f"\nHidden Layer Sizes: {hidden_layer_sizes}"
#     f"\nSolver: {solver}"
#     f"\nLearning Rate: {learning_rate}"
#     f"\nAlpha: {alpha}\n")
# print(ml_settings)
# model_fit, model_runtime = mlop.run_dnn(preprocessor, train_X, train_y, 
#                                         hidden_layer_sizes, solver, alpha, 
#                                         learning_rate, model_name,
#                                         activation_function,max_iter)

# =============================================================================
                                                                # Model Testing

model_stats = mlop.compute_predictive_performance(test_data, test_X, model_fit, 
                                                  model_name)
plt.rcdefaults()
model_plot = mlop.plot_model_performance(model_stats, model_runtime, 
                                          security_tag)
end_time = time.time()
end_tag_datetime = datetime.fromtimestamp(end_time)
end_tag = str(end_tag_datetime.strftime('%d%m%Y-%H%M%S'))
outputs_path = os.path.join('outputs',end_tag)
os.makedirs(outputs_path, exist_ok=True)
total_runtime = int(end_time - start_time)
model_plot.save(filename = f'{end_tag}.png',
                path = outputs_path,
                dpi = 600)
csv_path = os.path.join(outputs_path,f"{end_tag}.csv")
dataset.to_csv(csv_path)

end_tag_format = str(end_tag_datetime.strftime('%c'))
end_time_format = f"\n{end_tag_format}"
print(end_time_format)
total_model_runtime = f"\nTotal model runtime: {str(total_runtime)} seconds"
print(total_model_runtime)

txt_path = os.path.join(outputs_path, f"{end_tag}.txt")
with open(txt_path, 'w') as file:
    file.write(str(n_prices))
    file.write(str(excluded_file_format))
    file.write(str(start_tag_format))
    file.write(str(model_settings))
    file.write(str(activation_function_tag))
    file.write(str(max_iter_tag))
    file.write(str(model_name))
    file.write(str(end_time_format))
    file.write(str(total_model_runtime))
