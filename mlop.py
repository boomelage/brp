#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:24:36 2024

"""

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, facet_wrap, labs, theme
from time import time
import matplotlib.pyplot as plt

class mlop:
    
    '''
    Machine Learning Option Pricing with sklearn
    
    Parameters:
        model_scaler1
        random_state
        activation_function
        max_iter
        test_size
        rf_n_estimators
        rf_min_samples_leaf
        target_name
        security_tag
        feature_set
        user_dataset
    '''
    def __init__(self,
                 random_state,
                 max_iter,
                 test_size,
                 hidden_layer_sizes,
                 solver,
                 alpha,
                 learning_rate,
                 rf_n_estimators,
                 rf_min_samples_leaf,
                 target_name,
                 security_tag,
                 feature_set,
                 user_dataset,
                 transformers,
                 model_scaler1,
                 model_scaler2):
        self.random_state = random_state
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.rf_n_estimators = rf_n_estimators
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.target_name = target_name
        self.security_tag = security_tag
        self.feature_set = feature_set
        self.user_dataset = user_dataset
        self.transformers = transformers
        self.model_scaler1 = model_scaler1
        self.model_scaler2 = model_scaler2
# =============================================================================
                                                                # Preprocessing

    def split_user_data(self):
        train_data, test_data = train_test_split(
            self.user_dataset, 
            test_size=self.test_size, 
            random_state=self.random_state)
         
        train_X = train_data[self.feature_set]
        test_X = test_data[self.feature_set]
        
        train_y = train_data[self.target_name]
        test_y = test_data[self.target_name]
        
        return train_data, train_X, train_y, \
            test_data, test_X, test_y
            
    def preprocess(self):
        preprocessor = ColumnTransformer(transformers=self.transformers)
        print(f"Data Processed with the {str(self.model_scaler1)[:-2]}"
              f"{str(self.model_scaler2)}")
        return preprocessor
# =============================================================================
                                                             # Model Estimation

    def run_nnet(
            self, preprocessor, train_X, train_y, model_name, solver, 
            hidden_layer_sizes, activation_function, max_iter, random_state):
        print(model_name)
        nnet_start = time()
        nnet_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes[0],
            activation=activation_function, 
            solver=solver, 
            max_iter=max_iter,
            random_state=random_state
            )
            
        nnet_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", nnet_model)
            ])
        
        model_fit = nnet_pipeline.fit(train_X, train_y)
        nnet_end = time()
        nnet_runtime = int(nnet_end - nnet_start)
        print(f"Single Layer Network estimated in {str(nnet_runtime)} "
              "seconds!")
        return model_fit, nnet_runtime
    
    def run_dnn(self, preprocessor, train_X, train_y, hidden_layer_sizes,
                solver, alpha, learning_rate, model_name, activation_function, 
                max_iter):
        print(f"{str(model_name)} ({activation_function} activation)")
        dnn_start = time()
        deepnnet_model = MLPRegressor(
            hidden_layer_sizes= hidden_layer_sizes,
            activation = activation_function, 
            solver= solver,
            alpha = alpha,
            learning_rate = learning_rate,
            max_iter = max_iter, 
            random_state = self.random_state
            )
                                  
        deepnnet_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", deepnnet_model)
        ])

        dnn_fit = deepnnet_pipeline.fit(train_X,train_y)
        dnn_end = time()
        dnn_runtime = int(dnn_end - dnn_start)
        print(f"Deep Neural Network estimated in {str(dnn_runtime)} seconds!")
        return dnn_fit, dnn_runtime
    
# =============================================================================
                                                                # Model Testing
                                                                
    def compute_predictive_performance(self, test_data, test_X, model_fit, 
                                       model_name):
        predictive_performance = (pd.concat(
            [test_data.reset_index(drop=True), 
             pd.DataFrame({model_name: model_fit.predict(test_X)})
            ], axis=1)
          .melt(
            id_vars=self.user_dataset.columns,
            var_name="Model",
            value_name="Predicted"
          )
          .assign(
            moneyness=lambda x: x["spot_price"]*100/x["strike_price"],
            pricing_error=lambda x: 
                np.abs(abs(x["Predicted"] - \
                           x[self.target_name])*100/x[self.target_name])
          )
        )
        predictive_performance = predictive_performance.iloc[:,1:]
        return predictive_performance
    
    def plot_model_performance(self, predictive_performance, runtime, 
                               security_tag):
        predictive_performance_plot = (
            ggplot(predictive_performance, 
                   aes(x="moneyness", y="pricing_error")) + 
            geom_point(alpha=0.05) + 
            facet_wrap("Model") + 
            labs(x="Percentage moneyness (S/K)", 
                 y=f"Absolute percentage error ({runtime} second runtime)",
                 title=f'Prediction error for {security_tag} under Heston') + 
            theme(legend_position="")
            )
        predictive_performance_plot.show()
        plt.cla()
        plt.clf()
        return predictive_performance_plot    
    
    
    