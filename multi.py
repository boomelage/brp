# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:58:02 2024

"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,\
    MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, \
        SplineTransformer, PolynomialFeatures, KernelCenterer
import time
from datetime import datetime
from mlop import mlop
import numpy as np
from plotnine import *
# =============================================================================
                                                             # General Settings
random_state = 42
test_size = 0.01
                                                      # Neural Network Settings
max_iter = 100
hidden_layer_sizes=(10, 10, 10)
solver= [
            # "lbfgs",
            "sgd",
            # "adam"
        ]
alpha = 0.0001 #can't be none
learning_rate = 'adaptive'
                                                       # Random Forest Settings
rf_n_estimators = 50
rf_min_samples_leaf = 2000
                                                               # Model Settings
target_name = 'observed_price'
security_tag = 'vanilla options'
feature_set = ['spot_price', 
               'strike_price',
               'risk_free_rate',
               'years_to_maturity',
               'volatility',
               'w',
               'dividend_rate',
               'kappa',
               'theta',
               'sigma',
               'rho',
               'v0']
dataset = pd.read_csv(r'heston_vanillas2.csv')
# =============================================================================
solver = solver[0]
start_time = time.time()
start_tag = datetime.fromtimestamp(time.time())
start_tag = start_tag.strftime('%d%m%Y-%H%M%S')
print(datetime.fromtimestamp(time.time()))
print("\nSelected Parameters:")
print("\nFeatures:")
for feature in feature_set:
    print(feature)
print(f"\nTarget: {target_name}")
print(f"\nSecurity: {security_tag}")
print(f"\nSolver: {solver}")

model_name = f"{hidden_layer_sizes} Deep Neural Network"
    
mlop = mlop(
    random_state=random_state,
    max_iter=max_iter,
    test_size=test_size,
    rf_n_estimators=rf_n_estimators,
    rf_min_samples_leaf=rf_min_samples_leaf,
    target_name=target_name,
    security_tag=security_tag,
    feature_set=feature_set,
    user_dataset=dataset,
    hidden_layer_sizes=hidden_layer_sizes,
    solver=solver,
    alpha=alpha,
    learning_rate=learning_rate
)

# Model 1
preprocessor, train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.process_user_data(test_size, 
                                                       random_state, 
                                                       PowerTransformer())   
mod1, model_runtime1 = mlop.run_dnn(preprocessor, train_X, train_y, 
                                    hidden_layer_sizes, solver, alpha, 
                                    learning_rate, model_name, 'relu', 
                                    max_iter)


# Model 2
preprocessor, train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.process_user_data(test_size, 
                                                       random_state, 
                                                       PowerTransformer())
mod2, model_runtime2 = mlop.run_dnn(preprocessor, train_X, train_y, 
                                    hidden_layer_sizes, solver, alpha, 
                                    learning_rate, model_name, 'tanh', 
                                    max_iter)


# Model 3
preprocessor, train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.process_user_data(test_size, 
                                                       random_state, 
                                                       SplineTransformer())
mod3, model_runtime3 = mlop.run_dnn(preprocessor, train_X, train_y, 
                                    hidden_layer_sizes, solver, alpha, 
                                    learning_rate, model_name, 
                                    'relu', max_iter)


# Model 4
preprocessor, train_data, train_X, train_y, \
    test_data, test_X, test_y = mlop.process_user_data(test_size, 
                                                       random_state, 
                                                       SplineTransformer())
mod4, model_runtime4 = mlop.run_dnn(preprocessor, train_X, train_y, 
                                    hidden_layer_sizes, solver, alpha, 
                                    learning_rate, model_name, 'tanh', 
                                    max_iter)
# =============================================================================
                                                                # Model Testing
predictive_performance = (pd.concat(
    [test_data.reset_index(drop=True), 
     pd.DataFrame({
         f'dnn relu power ({model_runtime1} seconds)': mod1.predict(test_X),
         f'dnn tanh power ({model_runtime2} seconds)': mod2.predict(test_X),
         f'dnn relu spline ({model_runtime3} seconds)': mod3.predict(test_X),
         f'dnn tanh spline ({model_runtime4} seconds)': mod4.predict(test_X)
         })
    ], axis=1)
  .melt(
    id_vars=test_data.columns,
    var_name="Model",
    value_name="Predicted"
  )
  .assign(
    moneyness=lambda x: x["spot_price"] - x["strike_price"],
    pricing_error=lambda x: 
        np.abs(x["Predicted"] - x[target_name])
  )
)
predictive_performance = predictive_performance.iloc[:,1:]


model_plot = (
  ggplot(predictive_performance, 
         aes(x="moneyness", y="pricing_error")) +
  geom_point(alpha=0.05) +
  facet_wrap("Model") + 
  labs(x="Moneyness (S - K)", y="Absolut prediction error",
       title=f"Prediction errors (10, 10, 10) neural networks ({solver} solver)") +
  theme(legend_position="")
)
model_plot.draw()
model_plot.show()
end_time = time.time()
end_tag = datetime.fromtimestamp(end_time)
end_tag = end_tag.strftime('%d%m%Y-%H%M%S')
model_plot.save(filename = f'dnn_{solver}_{end_tag}.png',
                path = r"E:\OneDrive - rsbrc\Files\Dissertation",
                dpi = 600)
print(datetime.fromtimestamp(end_time))