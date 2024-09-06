#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:39:41 2024

"""
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,\
    MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, \
        SplineTransformer, PolynomialFeatures, KernelCenterer, \
            QuantileTransformer
from data_generation import generate_dataset, spotmin, spotmax, nspots, \
    n_maturities, n_strikes, lower_moneyness, upper_moneyness, \
        shortest_maturity, longest_maturity
import time
import textwrap
from datetime import datetime
from mlop import mlop

# =============================================================================
                                                             # General Settings
model_scaler = [
                StandardScaler(),
                # QuantileTransformer(),
                # MaxAbsScaler(),
                # MinMaxScaler(),
                # RobustScaler(),
                # Normalizer(),
                # PowerTransformer(),
                # SplineTransformer(),
                # PolynomialFeatures(),
                # KernelCenterer()
                ]
random_state = None

test_size = 0.05

                                                      # Neural Network Settings
max_iter = 1000
activation_function = [        
    # 'identity', 
    # 'logistic',
    'tanh',
    # 'relu',
    ]
hidden_layer_sizes=(10, 10, 10)
solver= [
            # "lbfgs",
            "sgd",
            # "adam"
        ]
alpha = 0.0001 #can't be none
learning_rate = 'adaptive'

                                                       # Random Forest Settings
rf_n_estimators = 50
rf_min_samples_leaf = 2000

                                                               # Model Settings
target_name = 'observed_price'
security_tag = 'vanilla options'
feature_set = [
    'spot_price',
    'strike_price',
    'years_to_maturity',
    # 'volatility',
    # 'risk_free_rate',
    # 'dividend_rate',
    # 'kappa',
    # 'theta',
    # 'sigma',
    # 'rho',
    # 'v0'
    ]

start_time = time.time()
start_tag = datetime.fromtimestamp(time.time())
start_tag = start_tag.strftime('%d%m%Y-%H%M%S')

dataset = generate_dataset()

print(f'\nNumber of option price/parameter sets generated: {len(dataset)}')


# =============================================================================
                                                                 # Loading mlop
activation_function = activation_function[0]
solver = solver[0]
model_scaler = model_scaler[0]
model_scaler_str = str(model_scaler)[:-2]
mlop = mlop(
    random_state=random_state,
    max_iter=max_iter,
    test_size=test_size,
    rf_n_estimators=rf_n_estimators,
    rf_min_samples_leaf=rf_min_samples_leaf,
    target_name=target_name,
    security_tag=security_tag,
    feature_set=feature_set,
    user_dataset=dataset,
    hidden_layer_sizes=hidden_layer_sizes,
    solver=solver,
    alpha=alpha,
    learning_rate=learning_rate
)
feature_str_list = '\n'.join(feature_set)
model_settings = (
    f"\n{datetime.fromtimestamp(time.time())}\n\nSelected Parameters:\n\nFeatures:"
    f"\n{feature_str_list}\n\nTarget: {target_name}\n\nSecurity: {security_tag}\n"
    )
print(model_settings)

# =============================================================================
                                                           # Preprocessing Data                                                 
preprocessor, train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.process_user_data(test_size, random_state, 
                                                       model_scaler)

# =============================================================================
                                                              # Model Selection

# print(f'Activation function: {activation_function}')
# print(f'Maximum iterations: {max_iter}')      
# model_name = f"Single Layer Network ({model_scaler_str})"                           
# model_fit, model_runtime = mlop.run_nnet(preprocessor, train_X, 
#                                           train_y, model_name)

model_name = f"{hidden_layer_sizes} Deep Neural Network "\
f"({activation_function}) ({model_scaler_str}) ({solver})"


ml_settings = (
    f"\n{datetime.fromtimestamp(time.time())}\n\nSelected Parameters:\n"
    f"\nScaler: {model_scaler_str}"
    f"\nActivation function: {activation_function}"
    f"\nMaximum iterations: {max_iter}"
    f"\nHidden Layer Sizes: {hidden_layer_sizes}"
    f"\nSolver: {solver}"
    f"\nLearning Rate: {learning_rate}"
    f"\nAlpha: {alpha}\n"
)
print(ml_settings)
model_fit, model_runtime = mlop.run_dnn(preprocessor, train_X, train_y, 
                                        hidden_layer_sizes, solver, alpha, 
                                        learning_rate, model_name,
                                        activation_function,max_iter)

# print(f"Random Forest Estimators: {rf_n_estimators}")
# print(f"Random Forest Min Samples Leaf: {rf_min_samples_leaf}")   

# =============================================================================
                                                                # Model Testing
                                                                
model_stats = mlop.compute_predictive_performance(test_data, test_X, model_fit, 
                                                  model_name)
print(model_stats)
model_plot = mlop.plot_model_performance(model_stats, model_runtime, 
                                          security_tag)

end_time = time.time()

end_tag = datetime.fromtimestamp(end_time)
end_tag = str(end_tag.strftime('%d%m%Y-%H%M%S'))
outputs_path = os.path.join('outputs',end_tag)
os.makedirs(outputs_path, exist_ok=True)
total_runtime = int(end_time - start_time)
model_plot.save(filename = f'{end_tag}.png',
                path = outputs_path,
                dpi = 600)
model_plot.show()
csv_path = os.path.join(outputs_path,f"{end_tag}.csv")
dataset.to_csv(csv_path)
print(f"\n{datetime.fromtimestamp(end_time)}")
total_model_runtime = f"Total model runtime: {str(total_runtime)} seconds"
print(f"{total_model_runtime}\n")
output = f"""Model estimated using {len(dataset)} options
with {nspots} spot price(s) between {spotmin} and {spotmax} (mid-point if one),
{n_strikes} strike(s) between {int(lower_moneyness*100)}% and
{int(upper_moneyness*100)}% moneyness, and {n_maturities} maturity/maturities
between {round(shortest_maturity,2)} and {round(longest_maturity,2)} years
(act/365)"""
wrapped_output = textwrap.fill(output, width=70)
txt_path = os.path.join(outputs_path,f"{end_tag}.txt")
with open(txt_path, 'w') as file:
    file.write(total_model_runtime)
    file.write(" \n")
    file.write(wrapped_output)
    file.write(model_settings)
    file.write(ml_settings)

print(wrapped_output)


# =============================================================================

