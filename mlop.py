#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:24:36 2024

"""
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,\
    MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, \
        SplineTransformer, PolynomialFeatures, KernelCenterer, \
            QuantileTransformer, OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import Lasso
from plotnine import ggplot, aes, geom_point, labs, theme
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
class mlop:
    
    def __init__(self,user_dataset):
        self.user_dataset = user_dataset.copy().replace(0,np.nan).dropna()
        self.random_state = None
        self.test_size = 0.01
        self.max_iter = int(1e3)
        self.hidden_layer_sizes = (10,10,10)
        self.single_layer_size = 10
        self.solver = [
                    # "lbfgs",
                    "sgd", 
                    # "adam"
                    ]
        
        self.alpha = 0.0001
        self.learning_rate_init = 0.001
        self.learning_rate = [
            
            'adaptive',
            # 'constant'
            
            ]
        
        self.activation_function = [  
            
            # 'identity',
            # 'logistic',
            # 'tanh',
            'relu',
            
            ]
        
        
        self.rf_n_estimators = 50
        self.rf_min_samples_leaf = 2000
        
        self.target_list = [
            # 'heston_price',
            'observed_price'
            ]
        self.target_name = self.target_list[0]
        
        """
        [
         'spot_price', 'strike_price', 'w', 'heston_price', 
         '30D', '60D', '3M', '6M', '12M', '18M', '24M', 'moneyness',
         'risk_free_rate', 'dividend_rate',
         'kappa', 'theta', 'rho', 'eta', 'v0', 'days_to_maturity',
         'expiration_date', 'calculation_date', 'moneyness_tag',
         ]
        """
        
        self.numerical_features = [
            'spot_price', 'strike_price', 'days_to_maturity', 
            # '30D', '60D', '3M', '6M', '12M', '18M', '24M',
            # 'risk_free_rate', 
            # 'dividend_rate',
            # 'moneyness', 
            # 'kappa', 'theta', 'eta', 'rho', 'v0',  
            ]
        
        self.categorical_features = [
            
            # 'barrier_type_name',
            
            # 'outin',
            
            # 'updown',
            
            # 'moneyness_tag',
            
            # 'calculation_date', 'expiration_date',
            
            # 'w'
            
            ]
        self.feature_set = self.numerical_features + self.categorical_features
        
        self.transformers = [
            # ("QuantileTransformer",QuantileTransformer(),self.numerical_features),
            ("StandardScaler",StandardScaler(),self.numerical_features),
            # ("MinMaxScaler",MinMaxScaler(),self.numerical_features),
            # ("MaxAbsScaler",MaxAbsScaler(),self.numerical_features),
            # ("PowerTransformer",PowerTransformer(),self.numerical_features),
            # ("Normalizer",Normalizer(),self.numerical_features),
            
            # ("OrdinalEncoder", OrdinalEncoder(),self.categorical_features),
            # ("OneHotEncoder", OneHotEncoder(),self.categorical_features)

            ]
        
        self.target_transformer_pipeline = Pipeline([
                ("StandardScaler", StandardScaler()),
                # ("RobustScaler", RobustScaler()),
                ])
        
        self.activation_function = self.activation_function[0]
        self.learning_rate = self.learning_rate[0]
        self.solver = self.solver[0]
        print(f"test size: {round(self.test_size*100,0)}%")
        print(f"random state: {self.random_state}")
        print(f"maximum iterations: {self.max_iter}")
        print(f"\ntarget: \n{self.target_name}")
        print(f"\nfeatures: \n{self.feature_set}")
        print("\nfeature transformer(s):")
        for i in self.transformers:
            print(f"{i}\n")
        print("target transformer(s):")
        for i in self.target_transformer_pipeline:
            print(f"{i}")
    """            
    ===========================================================================
    preprocessing
    """
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
        preprocessor = ColumnTransformer(
            transformers=self.transformers)
        return preprocessor
    
    """
    ===========================================================================
    model estimation
    """
    def run_nnet(self, preprocessor, train_X, train_y):
        model_name = "Single Layer Network"
        print(f"\n{model_name}")
        print(f"hidden layer size: {self.single_layer_size}")
        print(f"learning rate: {self.learning_rate}")
        print(f"activation: {self.activation_function}")
        print(f"solver: {self.solver}")
        print(f"alpha: {self.alpha}")
        print('\ntraining...')
        nnet_start = time.time()
        
        nnet_model = MLPRegressor(
            hidden_layer_sizes=self.single_layer_size,
            activation=self.activation_function,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
            )
            
        nnet_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", nnet_model)
            ])
        
        nnet_scaled = TransformedTargetRegressor(
            regressor=nnet_pipeline,
            transformer=self.target_transformer_pipeline 
        )
        
        model_fit = nnet_scaled.fit(train_X, train_y)
        nnet_end = time.time()
        nnet_runtime = int(nnet_end - nnet_start)
        return model_fit, nnet_runtime
    
    def run_dnn(self, preprocessor,train_X,train_y):
        model_name = "Deep Neural Network"
        print(f"\n{model_name}")
        print(f"hidden layers sizes: {self.hidden_layer_sizes}")
        print(f"learning rate: {self.learning_rate}")
        print(f"activation: {self.activation_function}")
        print(f"solver: {self.solver}")
        print(f"alpha: {self.alpha}")
        print('\ntraining...')
        dnn_start = time.time()
        deepnnet_model = MLPRegressor(
            hidden_layer_sizes= self.hidden_layer_sizes,
            activation = self.activation_function, 
            solver= self.solver,
            alpha = self.alpha,
            learning_rate = self.learning_rate,
            max_iter = self.max_iter, 
            random_state = self.random_state,
            learning_rate_init=self.learning_rate_init
            )
                                  
        dnn_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", deepnnet_model)
        ])
        
        dnn_scaled = TransformedTargetRegressor(
            regressor=dnn_pipeline,
            transformer=self.target_transformer_pipeline 
        )
        
        model_fit = dnn_scaled.fit(train_X,train_y)
        dnn_end = time.time()
        dnn_runtime = int(dnn_end - dnn_start)
        return model_fit, dnn_runtime
    
    def run_rf(self, preprocessor, train_X, train_y):
        model_name = "Random Forest"
        print(f"\n{model_name}")
        print(f"number of estimators: {self.rf_n_estimators}")
        print(f"minimum samples per leaf: {self.rf_min_samples_leaf}")
        print('\ntraining...')
        rf_start = time.time()
        
        rf_model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators, 
            min_samples_leaf=self.rf_min_samples_leaf, 
            random_state=self.random_state,
        )
        
        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", rf_model)])
        
        rf_scaled = TransformedTargetRegressor(
            regressor=rf_pipeline,
            transformer=self.target_transformer_pipeline 
        )
        
        model_fit = rf_scaled.fit(train_X, train_y)
        
        rf_end = time.time()
        rf_runtime = rf_end - rf_start
        estimation_end_tag = str(datetime.fromtimestamp(
            rf_end).strftime("%Y-%m-%d %H%M%S"))
        joblib.dump(model_fit, str(f"{model_name} {estimation_end_tag}.pkl"))
        return model_fit, rf_runtime
    
    def run_lm(self, train_X, train_y):
        model_name = "Lasso Regression"
        print(f"\n{model_name}")
        print(f"alpha: {self.alpha}")
        print('\ntraining...')
        lm_start = time.time()
        lm_pipeline = Pipeline([
            ("polynomial", PolynomialFeatures(degree=5, 
                                    interaction_only=False, 
                                    include_bias=True)),
            ("scaler", StandardScaler()),
            ("regressor", Lasso(alpha=self.alpha))])
        
        lm_scaled = TransformedTargetRegressor(
            regressor=lm_pipeline,
            transformer=self.target_transformer_pipeline 
        )
        
        model_fit = lm_scaled.fit(train_X, train_y)
        
        lm_end = time.time()
        lm_runtime = lm_end - lm_start
        return model_fit, lm_runtime


    """
    ===========================================================================
    standard model testing
    """
    
    def test_model(self,test_data,test_X,test_y,model_fit):
        training_results = test_X.copy()
        training_results['moneyness'] = test_data.loc[test_X.index,'moneyness']
        training_results['target'] = test_y
        training_results['prediciton'] = model_fit.predict(test_X)
        training_results['abs_relative_error'] = abs(
            training_results['prediciton']/training_results['target']-1)
        
        descriptive_stats = training_results['abs_relative_error'].describe()
        test_count = int(descriptive_stats['count'])
        descriptive_stats = descriptive_stats[1:]
        pd.set_option('display.float_format', '{:.10f}'.format)
        print(
            f"\nresults:\n--------\ntest data count: {test_count}"
            f"\n{descriptive_stats}\n"
            )
        pd.reset_option('display.float_format')
        
        return training_results

    def plot_model_performance(self, predictive_performance, runtime, title):
        predictive_performance_plot = (
            ggplot(predictive_performance, 
                   aes(x="moneyness", y="abs_relative_error")) + 
            geom_point(alpha=0.05) + 
            labs(x="relative moneyness", 
                 y=f"absolute relative error ({int(runtime)} second runtime)",
                 title=title) + 
            theme(legend_position="")
            )
        predictive_performance_plot.show()
        plt.cla()
        plt.clf()
        return predictive_performance_plot    



    