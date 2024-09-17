# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:08:31 2024

@author: boomelage






General process:
    
    1. collect market data
    
        option 1:
            collect term structure of volatility and apply Derman's method to 
            obtain slope coefficients of volatility against strike for every
            maturity available

        option 2:
            collect contract details directly and aggregate
            
    2. calibrate Heston model
    
        option 1: 
            for term structure data, use routine_calibration.py to calibrate 
            the model by maturitiy for all strikes in the volatility surface.
            This will yield a set of parameters for every maturity but only one
            strike.
            
        option 2:
            alternatively, one can use routine_collection.py to collect market
            data for even a few days at a time and calibrate by day. in this 
            approach, the data is collected and first grouped by spot_price.
            the calibration is then performed by maturity for all strikes. in 
            this process, the result will be a dataframe of all available comb-
            inations of parameter values for spot prices and maturities.
            
    3. generating large training dataset
        
        option 1:
            this option allows for the greatest amount of flexibility as the 
            we assume the implied volatility is a functional form of moneyness
            and maturity. the only constraints being the availability of Derman
            coefficients. the rest of the features can then be mapped
        
        option 2:
            in this alternative approach, the cartesian product is performed by
            spot price between maturities and strikes. we then 




















"""

