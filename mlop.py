#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:24:36 2024

"""

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,\
    MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, \
        SplineTransformer, PolynomialFeatures, KernelCenterer, \
            QuantileTransformer
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from plotnine import ggplot, aes, geom_point, facet_wrap, labs, theme
from time import time
import matplotlib.pyplot as plt

class mlop:
    
    '''
    Machine Learning Option Pricing with sklearn
    
    '''
    def __init__(self,user_dataset):
        
        self.random_state = 42
        self.test_size = 0.10
        self.max_iter = 1000
        self.hidden_layer_sizes = (10,10,10)
        self.solver = [
                    # "lbfgs",
                    "sgd",
                    # "adam"
                    ]
        
        self.alpha = 0.0001
        self.learning_rate = [
            
            'adaptive',
            # 'constant'
            
            ]
        
        self.activation_function = [  
            
            # 'identity',
            # 'logistic',
            'tanh',
            # 'relu',
            
            ]
        

        
        self.rf_n_estimators = 50
        self.rf_min_samples_leaf = 2000
        
        self.target_name = 'observed_price'
        self.feature_set = [
            
            'spot_price', 
            'strike_price', 
            'days_to_maturity', 
            'w',
            'v0',
            'kappa', 
            'theta', 
            'rho', 
            'sigma', 
            
            ]
        
        self.numerical_features = [
            
            'spot_price', 
            'strike_price', 
            'days_to_maturity', 
            'v0',
            'kappa', 
            'theta', 
            'rho', 
            'sigma', 
            
            ]
        
        self.categorical_features = [
            
            'w'
            
            ]
        
        self.transformers = [
            ("scale1",StandardScaler(),self.numerical_features),
            ("scale2",QuantileTransformer(),self.numerical_features),
            ("encode", OneHotEncoder(),self.categorical_features)
            ]   
        
        self.security_tag = 'vanilla options'
        self.user_dataset = user_dataset
        self.model_scaler = StandardScaler()
        self.activation_function = self.activation_function[0]
        self.learning_rate = self.learning_rate[0]
        self.solver = self.solver[0]

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
        return preprocessor
    
# =============================================================================
                                                             # Model Estimation

    def run_nnet(self, preprocessor, train_X, train_y):
        nnet_start = time()
        nnet_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation_function, 
            solver=self.solver, 
            max_iter=self.max_iter,
            random_state=self.random_state
            )
            
        nnet_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", nnet_model)
            ])
        
        model_fit = nnet_pipeline.fit(train_X, train_y)
        nnet_end = time()
        nnet_runtime = int(nnet_end - nnet_start)
        return model_fit, nnet_runtime
    
    def run_dnn(self, preprocessor,train_X,train_y):
        dnn_start = time()
        deepnnet_model = MLPRegressor(
            hidden_layer_sizes= self.hidden_layer_sizes,
            activation = self.activation_function, 
            solver= self.solver,
            alpha = self.alpha,
            learning_rate = self.learning_rate,
            max_iter = self.max_iter, 
            random_state = self.random_state
            )
                                  
        deepnnet_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", deepnnet_model)
        ])
        dnn_fit = deepnnet_pipeline.fit(train_X,train_y)
        dnn_end = time()
        dnn_runtime = int(dnn_end - dnn_start)
        return dnn_fit, dnn_runtime
    
    def run_rf(self, preprocessor, train_X, train_y):
        rf_model = RandomForestRegressor(
        n_estimators=self.rf_n_estimators, 
        min_samples_leaf=self.rf_min_samples_leaf, 
        random_state=self.random_state)
        
        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", rf_model)])
        rf_fit = rf_pipeline.fit(train_X, train_y)
        return rf_fit
    
    def run_lm(self, train_X, train_y):
        lm_pipeline = Pipeline([
            ("polynomial", PolynomialFeatures(degree=5, 
                                    interaction_only=False, 
                                    include_bias=True)),
            ("scaler", self.model_scaler),
            ("regressor", Lasso(alpha=0.01))])

        lm_fit = lm_pipeline.fit(train_X, train_y)
        return lm_fit

# =============================================================================
                                                                # Model Testing
                                                                
    def compute_predictive_performance(self,test_data,test_X,model_fit):
        predictive_performance = (pd.concat(
            [test_data.reset_index(drop=True), 
             pd.DataFrame({"model_name": model_fit.predict(test_X)})
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
    
    
    def plot_model_performance(self,predictive_performance):
        predictive_performance_plot = (
            ggplot(predictive_performance, 
                   aes(x="moneyness", y="pricing_error")) + 
            geom_point(alpha=0.05) + 
            facet_wrap("Model") + 
            labs(x="Percentage moneyness (S/K)", 
                 y="Absolute percentage error (addruntimeyo second runtime)",
                 title=f'Prediction error for {self.security_tag} under Heston') + 
            theme(legend_position="")
            )
        predictive_performance_plot.show()
        plt.cla()
        plt.clf()
        return predictive_performance_plot    
    
    
    