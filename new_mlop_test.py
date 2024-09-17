# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:13:18 2024

"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')


from routine_generation_market import dataset
dataset
from mlop import mlop
mlop = mlop(user_dataset=dataset)


train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.split_user_data()

   
preprocessor = mlop.preprocess()


# model_fit, runtime = mlop.run_nnet(preprocessor, train_X, train_y)


# model_fit, runtime = mlop.run_dnn(preprocessor,train_X,train_y)


model_fit = mlop.run_rf(preprocessor,train_X,train_y)


# model_fit = mlop.run_lm(train_X,train_y)


df = mlop.compute_predictive_performance(test_data,test_X,model_fit)

predictive_performance_plot = mlop.plot_model_performance(df)


