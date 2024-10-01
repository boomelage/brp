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



rf_model = RandomForestRegressor(
    n_estimators=mlop.rf_n_estimators, 
    min_samples_leaf=mlop.rf_min_samples_leaf, 
    random_state=mlop.random_state,
)


rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", rf_model)])

target_transformer_pipeline = Pipeline([
        ("robust_scaler", RobustScaler()), 
        # ("standard_scaler", StandardScaler())
        ])
    
rf_with_target_scaling = TransformedTargetRegressor(
    regressor=rf_pipeline,
    transformer=target_transformer_pipeline 
)

rf_fit = rf_with_target_scaling.fit(train_X, train_y)

prediction = rf_fit.predict(test_X)

avg = np.average(np.abs(prediction/test_y-1))

print(f"\n{avg*100}%")