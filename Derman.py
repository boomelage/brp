#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:52:50 2024

"""
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
from data_query import dirdatacsv
import numpy as np
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns',None)

class derman():
    def __init__(self, data_files=dirdatacsv()):
        self.data_files = data_files
        self.mats = []
        
    def retrieve_ts(self):
        
        file = self.data_files[0]
        ts = pd.read_csv(file)
        ts = ts.set_index(ts.iloc[:,0]).drop(columns = ts.columns[0])
        ts = ts.astype(float)
        ts.columns = ts.columns.astype(int)
        ts.index = ts.index.astype(int)
        ts = ts.loc[
            
            5540:5640,
            28:169
    
            ]
        ts = ts.loc[:, (ts != 0).any(axis=0)]
    
        ks = ts.index.tolist()
        mats = ts.columns.tolist()
        return ks, mats, ts
        
    def compute_derman_ivols(self,maturity):
        TSatmat = ts.loc[:,maturity]
        strikes = ts.index
        S = int(np.median(strikes))
        x = np.array(strikes - S,dtype=float)
        atmvol = np.median(TSatmat)
        y = np.array(TSatmat - atmvol,dtype=float)
        model = LinearRegression()
        x = x.reshape(-1,1)
        model.fit(x,y)
        b = model.coef_[0]
        alpha = model.intercept_
        derman_ivols = model.predict(x)/100
        derman_ivols = derman_ivols*b + alpha + atmvol
        return b, alpha, atmvol, derman_ivols
        
    def get_derman_coefs(self):
        derman_coefs = {}
        ks, mats, ts = self.retrieve_ts()
        for i, maturity in enumerate(mats):
            for j, k in enumerate(ks):
                b, alpha, atmvol, derman_ivols = self.compute_derman_ivols(maturity)
                derman_coefs[int(f"{maturity}")] = [b, alpha, atmvol]
        derman_coefs = pd.DataFrame(derman_coefs)
        derman_coefs['coef'] = ['b','alpha','atmvol']
        derman_coefs.set_index('coef',inplace = True)
        return derman_coefs
    
    def make_derman_surface(self):
        derman_surface = np.empty(len(mats),dtype=object)
        
        for i, maturity in enumerate(mats):
            b, alpha, atmvol, derman_ivols = self.compute_derman_ivols(maturity)
            derman_surface[i] = derman_ivols
        return derman_surface
        
    def make_derman_df(self, derman_surface):
        derman_df = pd.DataFrame(index=ks, columns=mats)
        for i, maturity in enumerate(mats):
            for j, k in enumerate(ks):
                try:
                    derman_df.loc[k,maturity] = derman_surface[i][j]
                except Exception as e:
                    print(f"error {e}")
                    print(f'i={i},j={j}')
        return derman_df
    
    def derman_ivols_for_market(self,df,derman_coefs):
        b = derman_coefs.loc['b']
        alpha = derman_coefs.loc['alpha']
        K = df['strike_price']
        S = df['spot_price']
        iv = df['atmiv']
        df['volatility'] = \
            iv + \
                alpha[df['days_to_maturity']] +\
                    b[df['days_to_maturity']]*(K-S)
        return df
                    
derman = derman()

ks, mats, ts = derman.retrieve_ts()

S = np.median(ks)

derman_coefs = derman.get_derman_coefs()

derman_surface = derman.make_derman_surface()

derman_df = derman.make_derman_df(derman_surface)



import QuantLib as ql
implied_vols_matrix = ql.Matrix(len(ks),len(mats),0)
for i, strike in enumerate(ks):
    for j, maturity in enumerate(mats):
        implied_vols_matrix[i][j] = derman_df.loc[strike,maturity]




# from settings import model_settings
# ms = model_settings()
# settings, ezprint = ms.import_model_settings()
# dividend_rate = settings['dividend_rate']
# risk_free_rate = settings['risk_free_rate']
# calculation_date = settings['calculation_date']
# day_count = settings['day_count']
# calendar = settings['calendar']
# flat_ts = settings['flat_ts']
# dividend_ts = settings['dividend_ts']

# expiration_dates = ms.compute_ql_maturity_dates(mats)
# S = np.median(ks)
# black_var_surface = ql.BlackVarianceSurface(
#     calculation_date, calendar,
#     expiration_dates, ks,
#     implied_vols_matrix, day_count)

# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize']=(15,7)
# plt.style.use("dark_background")
# from matplotlib import cm
# import numpy as np
# import os

# target_maturity_ivols = derman_df.loc[:,]
# fig, ax = plt.subplots()
# ax.plot(ks, target_maturity_ivols, label="Black Surface")
# ax.plot(ks, target_maturity_ivols, "o", label="Actual")
# ax.set_xlabel("Strikes", size=9)
# ax.set_ylabel("Vols", size=9)
# ax.legend(loc="upper right")
# fig.show()

# plot_maturities = np.array(mats,dtype=float)/365.25
# moneyness = np.array(ks,dtype=float)
# X, Y = np.meshgrid(plot_maturities, moneyness)
# Z = np.array(
#     [
#     black_var_surface.blackVol(x, y) for x, y in zip(X.flatten(), Y.flatten())
#       ])
# Z = Z.reshape(X.shape)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(
#     X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_xlabel("Maturities", size=9)
# ax.set_ylabel("Strikes", size=9)
# ax.set_zlabel("Implied Volatility", size=9)
# ax.view_init(elev=30, azim=-35)
# plt.show() 
# plt.cla()
# plt.clf()
      
