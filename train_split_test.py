# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:14:45 2024

@author: boomelage
"""

from train_main import training_data
from mlop import mlop
import pandas as pd
from datetime import datetime
dataset = training_data.copy()
from mlop import *
"""
# =============================================================================
                            preprocessing data
"""




feature_list = [
    'spot_price', 'strike_price', 'days_to_maturity', 'barrier', 
    'risk_free_rate', 'dividend_rate', 'kappa', 'theta', 'rho', 'eta', 'v0', 
    'barrier_type_name', 'w'
    ]

target_name = 'observed_price'

pd.set_option("display.max_columns",None)

dataset.describe()



    


transformers = [
    ("StandardScaler",StandardScaler(),[
        'spot_price', 'strike_price', 'days_to_maturity', 
        'barrier', 'risk_free_rate', 'dividend_rate', 
        'kappa', 'theta', 'rho', 'eta', 'v0'
        ]),
    ("OneHotEncoder", OneHotEncoder(
        sparse_output=False),['barrier_type_name', 'w'])]
    

preprocessor = ColumnTransformer(transformers=transformers)


dnn_start = time.time()
deepnnet_model = MLPRegressor(
    hidden_layer_sizes= (10,10,10,10,10),
    activation = 'relu', 
    solver= 'sgd',
    alpha = 0.0001,
    learning_rate = 'adaptive',
    max_iter = 1000, 
    random_state = None,
    learning_rate_init= 0.001
    )
                          
dnn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", deepnnet_model)
])

target_transformer_pipeline = Pipeline([
        ("StandardScaler", StandardScaler()),
        # ("RobustScaler", RobustScaler()),
        ])
dnn_scaled = TransformedTargetRegressor(
    regressor=dnn_pipeline,
    transformer=target_transformer_pipeline 
)

model_fit = dnn_scaled.fit(train_X,train_y)



"""
testing
"""
insample_prediction = model_fit.predict(train_X)
abserror = np.abs(insample_prediction - train_y)
sqerror = (insample_prediction - train_y)**2
insample_RSME = np.sqrt(np.average(sqerror))
insample_MAE = np.average(abserror)


outofsample_prediction = model_fit.predict(test_X)
abserror = np.abs(outofsample_prediction - test_y)
sqerror = (outofsample_prediction-test_y)**2
outofsample_RSME = np.sqrt(np.average(sqerror))
outofsample_MAE = np.average(abserror)

print(f"\ninsample:\nRSME: {insample_RSME}\nMAE: {insample_MAE}")
print(f"\n\noutofsample:\nRSME: {outofsample_RSME}\nMAE: {outofsample_MAE}")