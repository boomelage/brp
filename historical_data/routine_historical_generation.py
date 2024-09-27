# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:10:31 2024

generation routine
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir,'train_data'))
import pandas as pd
import numpy as np
import QuantLib as ql
from itertools import product
from routine_calibration_global import calibrate_heston
from bicubic_interpolation import make_bicubic_functional, bicubic_vol_row
from settings import model_settings
ms = model_settings()
os.chdir(current_dir)
from routine_historical_collection import collect_historical_data

def generate_features(K,T,s):
    features = pd.DataFrame(
        product(
            [float(s)],
            K,
            T
            ),
        columns=[
            "spot_price", 
            "strike_price",
            "days_to_maturity"
                  ])
    return features



def generate_initial_barrier_features(s,T,K,outins,updown,ws):
    features = pd.DataFrame(
        product(
            [s],
            T,
            K,
            [updown],
            outins,
            ws
            ),
        columns=[
            'spot_price', 
            'days_to_maturity',
            'strike_price',
            'updown',
            'outin',
            'w'
                  ])
    return features


title = 'barrier options'

# T = ms.T
T = [10,30,90]


n_strikes = 5
down_k_spread = 0.05
up_k_spread = 0.05


n_barriers = 3
barrier_spread = 0.0010                   
n_barrier_spreads = 5


outins = [
    
    'Out',
    'In'
    
    ]


ws = [
      'call',
      'put'
      ]

"""
historical generation routine
"""
historical_data = collect_historical_data()

total = historical_data.shape[0]
historical_option_data = pd.DataFrame()

training_data = pd.DataFrame()
for i, row in historical_data.iterrows():
    s = row['spot_price']
    dtdate = row['date']
    calculation_date = ql.Date(dtdate.day,dtdate.month,dtdate.year)
    expiry_dates = np.array([
            calculation_date + ql.Period(30,ql.Days), 
            calculation_date + ql.Period(60,ql.Days), 
            calculation_date + ql.Period(3,ql.Months), 
            calculation_date + ql.Period(6,ql.Months),
            calculation_date + ql.Period(12,ql.Months), 
            # calculation_date + ql.Period(18,ql.Months), 
            # calculation_date + ql.Period(24,ql.Months)
          ],dtype=object)
    
    T = expiry_dates - calculation_date
    g = row['dividend_rate']
    """
    calibration dataset construction
    """

    n_hist_spreads = 10
    historical_spread = 0.005
    n_strikes = 10
    K = np.linspace(
        s*(1 - n_hist_spreads * historical_spread),
        s*(1 + n_hist_spreads * historical_spread),
        n_strikes)
    
    """
    NOTE: the make_bicubic_functional function takes the current volatility 
    surface estimated via Derman only for the coefficients
    """
    
    bicubic_vol = make_bicubic_functional(
        ms.derman_ts, 
        ms.derman_ts.index.tolist(), 
        ms.derman_ts.columns.tolist())
    
   
    calibration_dataset = generate_features(
        K, T, s)

    calibration_dataset = calibration_dataset.apply(
        bicubic_vol_row, axis = 1, bicubic_vol = bicubic_vol)
    calibration_dataset = calibration_dataset.copy()
    calibration_dataset['risk_free_rate'] = 0.04
    calibration_dataset['dividend_rate'] = g
    
    heston_parameters, performance_df = calibrate_heston(
        calibration_dataset, s, calculation_date)
    
    """"""
    
    
    K = np.linspace(s*0.9,s*1.1,n_strikes)
    
    """
    up features
    """

    initial_up_features = generate_initial_barrier_features(
        s,T,K,outins,'Up',ws)

    initial_up_features

    up_features = pd.DataFrame()

    for i, row in initial_up_features.iterrows():
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        barriers = np.linspace(
            s*(1+barrier_spread),
            s*(1+barrier_spread*n_barrier_spreads),
            n_barriers
            )    
        
        up_subset =  pd.DataFrame(
            product(
                [s],
                [t],
                [k],
                ['Up'],
                outins,
                ws,
                barriers
                ),
            columns=[
                'spot_price', 
                'days_to_maturity',
                'strike_price',
                'updown',
                'outin',
                'w',
                'barrier'
                      ])
        
        up_subset['barrier_type_name'] = up_subset['updown']+up_subset['outin']
        
        print(f"\n{up_subset}\n")

        up_features = pd.concat(
            [up_features, up_subset],
            ignore_index = True)    
            



    """
    down features
    """


    initial_down_features = generate_initial_barrier_features(
        s,T,K,outins,'Down',ws)

    initial_down_features

    down_features = pd.DataFrame()

    for i, row in initial_down_features.iterrows():
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        outin = row['outin']
        w = row['w']
        
        
        barriers = np.linspace(
            s*(1-barrier_spread*n_barrier_spreads),
            s*(1-barrier_spread),
            n_barriers
            )    
        
        down_subset =  pd.DataFrame(
            product(
                [s],
                [t],
                [k],
                ['Down'],
                [outin],
                [w],
                barriers
                ),
            columns=[
                'spot_price', 
                'days_to_maturity',
                'strike_price',
                'updown',
                'outin',
                'w',
                'barrier'
                      ])
        
        
        down_subset['barrier_type_name'] = down_subset['updown']+down_subset['outin']
        
        print(f"\n{down_subset}\n")
        
        down_features = pd.concat(
            [down_features, down_subset],
            ignore_index = True)    
            


    features = pd.concat([down_features,up_features],ignore_index=True)

    features['eta'] = heston_parameters['eta'].iloc[0]
    features['theta'] = heston_parameters['theta'].iloc[0]
    features['kappa'] = heston_parameters['kappa'].iloc[0]
    features['rho'] = heston_parameters['rho'].iloc[0]
    features['v0'] = heston_parameters['v0'].iloc[0]
    features['heston_price'] = np.nan
    features['heston_price'] = np.nan
    features['barrier_price'] = np.nan

    features
    pricing_bar = ms.make_tqdm_bar(
        desc="pricing",total=features.shape[0],unit='contracts')

    for i, row in features.iterrows():
        
        barrier_type_name = row['barrier_type_name']
        barrier = row['barrier']
        s = row['spot_price']
        k = row['strike_price']
        t = row['days_to_maturity']
        w = row['w']
        r = 0.04
        g = 0.001
        rebate = 0.
        
        v0 = heston_parameters['v0'].iloc[0]
        kappa = heston_parameters['kappa'].iloc[0] 
        theta = heston_parameters['theta'].iloc[0] 
        eta = heston_parameters['eta'].iloc[0] 
        rho = heston_parameters['rho'].iloc[0]
        
        
        black_scholes = ms.black_scholes_price(s,k,t,r,0.21,w)
        
        heston_price = ms.ql_heston_price(
            s,k,t,r,g,w,v0,kappa,theta,eta,rho,calculation_date
            )
        features.at[i,'heston_price'] = heston_price
        
        barrier_price = ms.ql_barrier_price(
                s,k,t,r,g,calculation_date,w,
                barrier_type_name,barrier,rebate,
                v0, kappa, theta, eta, rho)

        features.at[i,'barrier_price'] = barrier_price
        
        pricing_bar.update(1)
    pricing_bar.close()

    training_data = features.copy()

    training_data = ms.noisyfier(training_data)

    pd.set_option("display.max_columns",None)
    print(f'\n{training_data}\n')
    print(f'\n{training_data.describe()}\n')
    pd.reset_option("display.max_columns")

    historical_option_data = pd.concat(
        [historical_option_data, training_data],
        axis=1)




    
   