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
    # StandardScaler(),
    
    ""
    ]

transformers=[
    ("transformation_1",model_scaler[0],feature_set),
    # ("transformation_2", model_scaler[1],feature_set)
    ]     

random_state = 42
test_size = 0.01

                                                      # Neural Network Settings
max_iter = 10000
activation_function = [        
    'identity',
    # 'logistic',
    # 'tanh',
    # 'relu',
    ]
hidden_layer_sizes=(100, 100, 100)
solver= [
            "lbfgs",
            # "sgd",
            # "adam"
        ]
alpha = 0.0001
learning_rate = 'adaptive'

                                                       # Random Forest Settings
rf_n_estimators = 50
rf_min_samples_leaf = 2000

# =============================================================================

start_time = time.time()
start_tag = datetime.fromtimestamp(time.time())
start_tag_format = f"\n{str(start_tag.strftime('%c'))}"
start_tag = start_tag.strftime('%d%m%Y-%H%M%S')
print(start_tag_format)

# =============================================================================
                                                                 # loading data

# from routine_collection import collect_market_data_and_price
# excluded_file = r'SPXts.xlsx'
# ticker = excluded_file[:excluded_file.find('ts')]
# excluded_file_format = f"\nTerm sturcutre: {excluded_file}"
# print(excluded_file)
# dataset = collect_market_data_and_price(excluded_file)
# n_prices = f"\nestimated with {str(len(dataset))} "\
#     f"option prices collected from the market"
# print(n_prices)

from routine_generation import dataset
excluded_file = r'SPXts.xlsx'
ticker = excluded_file[:excluded_file.find('ts')]
excluded_file_format = f"\nTerm sturcutre: {excluded_file}"
n_prices = f"estimated with {len(dataset)} synthesized option prices"
print(f"\n{str(n_prices)}")

# import pandas as pd
# dataset = pd.read_csv(r"E:\git\brp\07092024-072748.csv")

# =============================================================================
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
    f"\nSelected Parameters:\n\nFeatures:\n{feature_str_list}\n\nTarget: "
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
model_name = f"Single Layer Network ({scaler1name}{scaler2name})"                           
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

plt.rcdefaults()
model_stats = mlop.compute_predictive_performance(test_data, test_X, model_fit, 
                                                  model_name)
model_plot = mlop.plot_model_performance(model_stats, model_runtime, 
                                          security_tag)
end_time = time.time()
end_tag_datetime = datetime.fromtimestamp(end_time)
end_tag = str(end_tag_datetime.strftime('%d%m%Y-%H%M%S'))
output_path_tag = str(f"{ticker} {end_tag}")
outputs_path = os.path.join('outputs',output_path_tag)
os.makedirs(outputs_path, exist_ok=True)
total_runtime = int(end_time - start_time)
model_plot.save(filename = f'{ticker} prediction {end_tag}.png',
                path = outputs_path,
                dpi = 600)
csv_path = os.path.join(outputs_path,f"{ticker} {end_tag}.csv")
dataset.to_csv(csv_path)
end_tag_format = str(end_tag_datetime.strftime('%c'))
end_time_format = f"\n{end_tag_format}"
print(end_time_format)
total_model_runtime = f"\nTotal model runtime: {str(total_runtime)} seconds"
print(total_model_runtime)
txt_path = os.path.join(outputs_path, f"{ticker} {end_tag}.txt")
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
    




# =============================================================================
                                  # plotting implied volatility for current day
# =============================================================================


from settings import model_settings
model_settings = model_settings()
settings = model_settings.import_model_settings()



dividend_rate = settings['dividend_rate']
risk_free_rate = settings['risk_free_rate']
calculation_date = settings['calculation_date']
day_count = settings['day_count']
calendar = settings['calendar']
flat_ts = settings['flat_ts']
dividend_ts = settings['dividend_ts']

import QuantLib as ql
from routine_calibration import implied_vol_matrix, strikes, maturities
expiration_dates = model_settings.compute_ql_maturity_dates(maturities)
black_var_surface = ql.BlackVarianceSurface(
    calculation_date, calendar,
    expiration_dates, strikes,
    implied_vol_matrix, day_count)


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm
import numpy as np
import os

# target_maturity_ivols = ivoldf[1]
# fig, ax = plt.subplots()
# ax.plot(strikes, target_maturity_ivols, label="Black Surface")
# ax.plot(strikes, target_maturity_ivols, "o", label="Actual")
# ax.set_xlabel("Strikes", size=9)
# ax.set_ylabel("Vols", size=9)
# ax.legend(loc="upper right")
# fig.show()

plot_maturities = np.array(maturities,dtype=float)/365.25
moneyness = np.array(strikes,dtype=float)
X, Y = np.meshgrid(plot_maturities, moneyness)
Z = np.array([black_var_surface.blackVol(x, y) for x, y in zip(X.flatten(), Y.flatten())])
Z = Z.reshape(X.shape)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Maturities", size=9)
ax.set_ylabel("Strikes", size=9)
ax.set_zlabel("Implied Volatility", size=9)
ax.view_init(elev=30, azim=-35)
plt.show()
plt.cla()
plt.clf()

# plot_volatility_surface(
#     outputs_path, ticker, ivoldf,strikes,maturities,black_var_surface)
# plt.rcdefaults()


