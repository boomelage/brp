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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
class mlop:
    
    def __init__(self,user_dataset):
        self.user_dataset = user_dataset.copy()
        self.random_state = None
        self.test_size = 0.01
        self.max_iter = int(1e3)
        self.hidden_layer_sizes = (26,26,26)
        self.single_layer_size = 26
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
        
        self.target_name = 'observed_price'
        
        """
        [
         'spot_price', 'strike_price', 'w', 'heston_price', 
         'risk_free_rate', 'dividend_rate', 'moneyness'
         'kappa', 'theta', 'rho', 'eta', 'v0', 'days_to_maturity',
         'expiration_date', 'calculation_date', 'moneyness_tag',
         ]
        """
        
        self.numerical_features = [
            'spot_price', 'strike_price', 'days_to_maturity', 
            'risk_free_rate',
            'dividend_rate',
            'kappa', 'theta', 'rho', 'eta', 'v0',
            
            'barrier',
            
            ]
        
        self.categorical_features = [
            
            'barrier_type_name',
            
            # 'outin',
            
            # 'updown',
            
            
            'w'
            
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
            ("OneHotEncoder", OneHotEncoder(
                sparse_output=False),self.categorical_features)
            
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
            print(i)
        print()
    """            
    ===========================================================================
    preprocessing
    """
    def split_user_data(self):
        train_data, test_data = train_test_split(
            self.user_dataset,
            test_size=self.test_size,
            random_state=self.random_state
            )
         
        train_X = train_data[self.feature_set]
        test_X = test_data[self.feature_set]
        
        train_y = train_data[self.target_name]
        test_y = test_data[self.target_name]
        
        return train_data, train_X, train_y, \
            test_data, test_X, test_y
            
    def split_data_manually(self,
            train_data, test_data,
            feature_set=None, target_name=None
            ):
        
        if feature_set == None:
            feature_set = self.feature_set
        if target_name == None:
            target_name = self.target_name

        
        test_X = test_data[feature_set]
        test_y = test_data[target_name]
        
        train_X = train_data[feature_set]
        train_y = train_data[target_name]
        return train_X, train_y, test_X, test_y

    def preprocess(self):
        preprocessor = ColumnTransformer(
            transformers=self.transformers)
        return preprocessor
    
    """
    ===========================================================================
    model estimation
    """
    def run_nnet(self, preprocessor, train_X, train_y):
        specs = [
            "Single Layer Network",
            f"hidden layer size: {self.single_layer_size}",
            f"learning rate: {self.learning_rate}",
            f"activation: {self.activation_function}",
            f"solver: {self.solver}",
            f"alpha: {self.alpha}"
        ]
        for spec in specs:
            print(spec)
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
        return model_fit, nnet_runtime, specs
    
    def run_dnn(self, preprocessor,train_X,train_y):
        specs= [
            "Deep Neural Network",
            f"hidden layers sizes: {self.hidden_layer_sizes}",
            f"learning rate: {self.learning_rate}",
            f"activation: {self.activation_function}",
            f"solver: {self.solver}",
            f"alpha: {self.alpha}"
            ]
        print('\ntraining...\n')
        for spec in specs:
            print(spec)
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
        return model_fit, dnn_runtime, specs
    
    def run_rf(self, preprocessor, train_X, train_y):
        specs = ["Random Forest",
        f"number of estimators: {self.rf_n_estimators}",
        f"minimum samples per leaf: {self.rf_min_samples_leaf}"]
        print('\ntraining...')
        for spec in specs:
            print(spec)
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
        return model_fit, rf_runtime, specs
    
    def run_lm(self, train_X, train_y):
        specs = ["Lasso Regression",f"alpha: {self.alpha}"]
        print('\ntraining...')
        for spec in specs:
            print(spec)
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
        return model_fit, lm_runtime, specs


    """
    ===========================================================================
    standard model testing
    """
    
    def test_prediction_accuracy(
            self,
            model_fit,
            test_data,
            train_data
            ):
        train_X = train_data[self.feature_set]
        train_y = train_data[self.target_name]
        test_X = test_data[self.feature_set]
        test_y = test_data[self.target_name]
        
        insample_prediction = model_fit.predict(train_X)
        insample_diff = insample_prediction - train_y
        insample_RMSE = np.sqrt(np.average(insample_diff**2))
        insample_MAE = np.average(np.abs(insample_diff))
        
        outofsample_prediction = model_fit.predict(test_X)
        outofsample_diff = outofsample_prediction-test_y
        outofsample_RMSE = np.sqrt(np.average(outofsample_diff**2))
        outofsample_MAE = np.average(np.abs(outofsample_diff))
        
        print("\nin sample:"
              f"\n     RSME: {insample_RMSE}"
              f"\n     MAE: {insample_MAE}")
        print("\nout of sample:"
              f"\n     RSME: {outofsample_RMSE}"
              f"\n     MAE: {outofsample_MAE}")
        
        insample_results = train_data.copy()
        insample_results['insample_target'] = train_y
        insample_results['insample_prediction'] = insample_prediction 
        insample_results['insample_error'] = insample_diff 
        
        outofsample_results = test_data.copy()
        outofsample_results['outofsample_target'] = test_y
        outofsample_results['outofsample_prediction'] = outofsample_prediction
        outofsample_results['outofsample_error'] = outofsample_diff
        
        errors = pd.Series(
            [
                insample_RMSE,insample_MAE,
                outofsample_RMSE,outofsample_MAE
                ],
            index=[
                'insample_RMSE','insample_MAE',
                'outofsample_RMSE','outofsample_MAE'],
            dtype=float
            )
        
        return insample_results, outofsample_results, errors
        
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

    def plot_model_performance(
            self, df, X_name, Y_name, xlabel, ylabel, runtime, title):
        predictive_performance_plot = (
            ggplot(df, 
                   aes(x=X_name, y=Y_name)) + 
            geom_point(alpha=0.05) + 
            labs(x=xlabel, 
                 y=ylabel,
                 title=title) + 
            theme(legend_position="")
            )
        predictive_performance_plot.show()
        plt.cla()
        plt.clf()
        return predictive_performance_plot    



    