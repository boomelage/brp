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
from data_generation import generate_dataset
import time
from datetime import datetime
from mlop import mlop

# =============================================================================
                                                             # General Settings
model_scaler = [
                # StandardScaler(),
                QuantileTransformer(),
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
max_iter = 10000
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
    'risk_free_rate',
    'years_to_maturity',
    # 'volatility',
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
spotmin = 95
spotmax = 105
nspots = 1000
tl_ivol = 0.374
spots = np.linspace(spotmin,spotmax,nspots)
dataset, tl_ivol = generate_dataset(spots,tl_ivol)
dataset.to_csv(f'{spotmin}-{spotmax}_tl_ivol{str(tl_ivol*100)}div10_{start_tag}')

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

# =============================================================================

dataset.to_csv(f'{start_tag}.csv')
print(f'\n{datetime.fromtimestamp(time.time())}')
print("\nSelected Parameters:")
print("\nFeatures:")
for feature in feature_set:
    print(feature)
print(f"\nTarget: {target_name}")
print(f"\nSecurity: {security_tag}")

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
print(f'\nScaler: {model_scaler_str}')
print(f'Activation function: {activation_function}')
print(f'Maximum iterations: {max_iter}')
print(f'Hidden Layer Sizes: {hidden_layer_sizes}')
print(f'Solver: {solver}')
print(f'Learning Rate: {learning_rate}')
print(f'Alpha: {alpha}')
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
end_tag = end_tag.strftime('%d%m%Y-%H%M%S')
total_runtime = int(end_time - start_time)

print(datetime.fromtimestamp(end_time))

print(f'\nModel estimated using {len(dataset)} options')
print(f'with {nspots} spot prices equidistantly spaced between {spotmin} to {spotmax}')

print(f'\nTotal model runtime: {str(total_runtime)} seconds')

# model_plot.save(filename = f'{end_tag}.png',
#                 path = r"E:\OneDrive - rsbrc\Files\Dissertation",
#                 dpi = 600)

print('DATA NOT ROUNDED!')

model_plot.show()

# =============================================================================

