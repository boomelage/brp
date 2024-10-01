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
from mlop import *
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


target_transformer_pipeline = Pipeline([
        ("scaler1", StandardScaler())
        ])
            
print(f"\ntarget transformations:\n{target_transformer_pipeline}\n")

"""
random forest with target scaling
"""

rf_model = RandomForestRegressor(
    n_estimators=mlop.rf_n_estimators, 
    min_samples_leaf=mlop.rf_min_samples_leaf, 
    random_state=mlop.random_state,
)


rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", rf_model)])

    
rf_scaled = TransformedTargetRegressor(
    regressor=rf_pipeline,
    transformer=target_transformer_pipeline 
)


rf_fit = rf_scaled.fit(train_X, train_y)

prediction = rf_fit.predict(test_X)

avg = np.average(np.abs(prediction/test_y-1))

print(f"\nrandom forest absolute average error: {round(avg*100,2)}%")


"""
deep netural network with target scaling
"""


deepnnet_model = MLPRegressor(
    hidden_layer_sizes= mlop.hidden_layer_sizes,
    activation = mlop.activation_function, 
    solver= mlop.solver,
    alpha = mlop.alpha,
    learning_rate = mlop.learning_rate,
    max_iter = mlop.max_iter, 
    random_state = mlop.random_state,
    learning_rate_init=mlop.learning_rate_init
    )

deepnnet_pipeline = Pipeline(
    [
     ("preprocessor", preprocessor),
     ("regressor", deepnnet_model)
     ]
    )

dnn_scaled = TransformedTargetRegressor(
    regressor=deepnnet_pipeline,
    transformer=target_transformer_pipeline 
)

dnn_fit = dnn_scaled.fit(train_X,train_y)

dnn_prediction = dnn_fit.predict(test_X)

dnn_avg = np.average(np.abs(dnn_prediction/test_y-1))

print(f"\ndeep neural network absolute average error: {round(dnn_avg*100,2)}%")



