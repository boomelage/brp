# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:28:49 2024

@author: boomelage
"""

import requests
import pandas as pd
from model_settings import ms
import numpy as np
import QuantLib as ql
import matplotlib.pyplot as plt
from matplotlib import cm
from model_settings import ms
date = '2009-04-01'
key = '9ZDGLN9SFCLWFN32'
symbol = 'SPY'
pd.set_option("display.max_columns",None)

underlying_url = str(
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
    f"{symbol}&outputsize=full&apikey={key}"
    )
spotr = requests.get(underlying_url)
spots = pd.DataFrame(spotr.json()['Time Series (Daily)']).T
spots = spots.astype(float)

spots['mid'] = np.array(
    (spots['3. low'].values + spots['2. high'].values)/2
    )
spots.index = pd.to_datetime(spots.index,format='%Y-%m-%d')

options_url = str(
    "https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&"
    f"symbol={symbol}"
    f"&date={date}"
    f"&apikey={key}"
          )

r = requests.get(options_url)
data = r.json()

df = pd.DataFrame(data['data'])
df = df.rename(
    columns={
        'expiration':'expiration_date',
        'date':'calculation_date',
        'strike':'strike_price',
        'type':'w',
        'implied_volatility':'volatility'
        }
    )

df =  df[
    [
     'strike_price','volatility','w','bid','ask',
     'calculation_date', 'expiration_date'
     ]
    ]

df['calculation_date'] = pd.to_datetime(df['calculation_date'])
df['expiration_date'] = pd.to_datetime(df['expiration_date'])
columns_to_convert = ['strike_price', 'volatility', 'bid', 'ask']
df[columns_to_convert] = df[
    columns_to_convert].apply(pd.to_numeric, errors='coerce')
df['mid'] = (df['bid'].values + df['ask'].values)/2


df['spot_price'] = df['calculation_date'].map(spots['mid'])
df['days_to_maturity'] = (
    df['expiration_date'] - df['calculation_date']).dt.days
df['moneyness'] = ms.vmoneyness(df['spot_price'], df['strike_price'], df['w'])



contracts = df[df['w']=='call'].copy().reset_index(drop=True)

contracts = contracts[np.abs(contracts['moneyness'])<0.10].copy()
contracts = contracts[np.abs(contracts['moneyness'])>0.05].copy()

s = float(contracts['spot_price'].unique()[0])
T = np.sort(contracts['days_to_maturity'].unique().astype(float)).tolist()
K = np.sort(contracts['strike_price'].unique().astype(float)).tolist()


contracts = contracts.set_index(['days_to_maturity','strike_price'])


ivol_df = pd.DataFrame(
    np.zeros((len(K),len(T)),dtype=float),
    index = K,
    columns = T
    )

for k in K:
    for t in T:
        try:
            ivol_df.loc[k,t] = contracts.loc[(t,k),'volatility']
        except Exception:
            ivol_df.loc[k,t] = np.nan


ivol_df = ivol_df.dropna(how='all',axis=1).dropna(how='all',axis=0)
ivol_df = ivol_df.fillna(0.0)


K = ivol_df.index.tolist()
T = ivol_df.columns.tolist()

ql_ivols = ql.Matrix(len(K),len(T),0.0)
for i,k in enumerate(K):
    for j,t in enumerate(T):
        ql_ivols[i][j] = float(ivol_df.loc[k,t])


bicubic_vol = ql.BicubicSpline(T, K, ql_ivols)


K = np.linspace(min(K),max(K),100)
T = np.linspace(min(T),360,100)
KK,TT = np.meshgrid(K,T)

V = np.array(
    [[bicubic_vol(float(t),float(k),False) for k in K] for t in T]
    )


plt.rcParams['figure.figsize']=(7,5)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=20, azim=30)  
surf = ax.plot_surface(KK,TT,V, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0.1)
fig.colorbar(surf, shrink=0.3, aspect=5)

ax.set_xlabel("Strikes", size=9)
ax.set_ylabel("Maturities (Years)", size=9)
ax.set_zlabel("Volatility", size=9)

plt.tight_layout() 
plt.show()
plt.cla()
plt.clf()
