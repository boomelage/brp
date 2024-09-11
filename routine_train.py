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

# from routine_collection import collect_market_data, collect_market_data_and_price
# excluded_file = r'SPXts.xlsx'
# dataset = collect_market_data_and_price(excluded_file)

from routine_generation import dataset


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
                # QuantileTransformer(),
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
    # ("transformation_2", model_scaler[1],feature_set)
    ]     

random_state = None
test_size = 0.05

                                                      # Neural Network Settings
max_iter = 1000
activation_function = [        
    # 'identity', 
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


start_time = time.time()
start_tag = datetime.fromtimestamp(time.time())
start_tag = start_tag.strftime('%d%m%Y-%H%M%S')





# =============================================================================

model_scaler1 = model_scaler[0]
model_scaler2 = model_scaler[1]
# Loading mlop
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
    f"\n{datetime.fromtimestamp(time.time())}\n\nSelected Parameters:"
    f"\n\nFeatures:\n{feature_str_list}\n\nTarget: {target_name}\n\nSecurity: "
    f"{security_tag}\n"
    )
print(model_settings)

# =============================================================================
                                                           # Preprocessing Data                                                 
train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.split_user_data()
    
    
preprocessor = mlop.preprocess()
# =============================================================================
                                                              # Model Selection
# print(f'Activation function: {activation_function}')
# print(f'Maximum iterations: {max_iter}')      
# model_name = f"Single Layer Network ({model_scaler_str})"                           
# model_fit, model_runtime = mlop.run_nnet(preprocessor, train_X, 
#                                           train_y, model_name)

model_name = f"{hidden_layer_sizes} Deep Neural Network "\
f"({activation_function}) ({scaler1name}{scaler2name}) ({solver})"
ml_settings = (
    f"\n{datetime.fromtimestamp(time.time())}\n\nSelected Parameters:\n"
    f"\nScaler: {scaler1name}{scaler2name}"
    f"\nActivation function: {activation_function}"
    f"\nMaximum iterations: {max_iter}"
    f"\nHidden Layer Sizes: {hidden_layer_sizes}"
    f"\nSolver: {solver}"
    f"\nLearning Rate: {learning_rate}"
    f"\nAlpha: {alpha}\n")
print(ml_settings)
model_fit, model_runtime = mlop.run_dnn(preprocessor, train_X, train_y, 
                                        hidden_layer_sizes, solver, alpha, 
                                        learning_rate, model_name,
                                        activation_function,max_iter)

# =============================================================================
                                                                # Model Testing
                                                                
model_stats = mlop.compute_predictive_performance(test_data, test_X, model_fit, 
                                                  model_name)
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
csv_path = os.path.join(outputs_path,f"{end_tag}.csv")
dataset.to_csv(csv_path)
print(f"\n{datetime.fromtimestamp(end_time)}")
total_model_runtime = f"Total model runtime: {str(total_runtime)} seconds"
print(f"{total_model_runtime}\n")

