```python
from model_settings import ms
import requests
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import QuantLib as ql
```

    
    pricing settings:
    Actual/365 (Fixed) day counter
    New York stock exchange calendar
    compounding: continuous
    frequency: annual
    
    


```python
key = ms.av_key
symbol = 'SPY'
date = '2024-05-17'
```


```python
underlying_url = str(
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
    f"&symbol={symbol}&date={date}&outputsize=compact&apikey={key}"
    )
spotr = requests.get(underlying_url)
spots = pd.DataFrame(spotr.json()['Time Series (Daily)']).T
spots = spots.astype(float)

spots.index = pd.to_datetime(spots.index,format='%Y-%m-%d')
spot = float(spots['4. close'][date])
spot
```




    529.45




```python
options_url = str(
    "https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&"
    f"symbol={symbol}"
    f"&date={date}"
    f"&apikey={key}"
          )

r = requests.get(options_url)
data = r.json()
```


```python
spot
```




    529.45




```python
df = pd.DataFrame(data['data'])
columns_to_convert = ['strike', 'last', 'mark',
       'bid', 'bid_size', 'ask', 'ask_size', 'volume', 'open_interest',
       'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']
df[columns_to_convert] = df[
    columns_to_convert].apply(pd.to_numeric, errors='coerce')

df['expiration'] = pd.to_datetime(df['expiration'],format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
df['days_to_maturity'] = df['expiration'] - df['date']
df['days_to_maturity'] = df['days_to_maturity'] / np.timedelta64(1, 'D')
df['days_to_maturity'] = df['days_to_maturity'].astype('int64')
df = df[(df['days_to_maturity']>=30)&(df['days_to_maturity']<=400)]

df = df[df['volume']>0].copy()

df['spot_price'] = spot
df['moneyness'] = ms.vmoneyness(df['spot_price'],df['strike'],df['type'])
df = df[(df['moneyness']<0)&(df['moneyness']>-0.5)]
indexed = df.copy().set_index(['strike','days_to_maturity'])
```


```python
plt.figure(figsize=(4,2))
plt.scatter(df['days_to_maturity'],df['volume'])
plt.title('volumes for maturities')
plt.show()
plt.figure(figsize=(4,2))
plt.scatter(df['strike'],df['volume'])
plt.title('volumes for maturities')
plt.show()
plt.clf()
```


    
![png](output_6_0.png)
    



    
![png](output_6_1.png)
    



    <Figure size 640x480 with 0 Axes>



```python
s = spot
T = np.sort(df['days_to_maturity'].unique()).tolist()
K = np.sort(df['strike'].unique()).tolist()
volume_heatmap = pd.DataFrame(np.full((len(K), len(T)), np.nan), index=K, columns=T)
for k in K:
    for t in T:
        try:
            volume_heatmap.loc[k,t] = indexed.loc[(k,t),'volume']
        except Exception:
            pass
```


```python
hottest = 50
```


```python
hottest_contracts = pd.DataFrame(volume_heatmap.unstack().sort_values(ascending=False)).head(hottest).reset_index()
hottest_contracts.columns = ['t','k','volume']
T = np.sort(hottest_contracts['t'].unique()).tolist()
K = np.sort(hottest_contracts['k'].unique()).tolist()

vol_matrix = pd.DataFrame(
    np.full((len(K),len(T)),np.nan),
    index = K,
    columns = T
)
for k in K:
    for t in T:
        try:
            vol_matrix.loc[k,t] = indexed.loc[(k,float(t)),'implied_volatility']
        except Exception:
            pass

vol_matrix = vol_matrix.dropna().copy()
T = vol_matrix.columns.tolist()
K = vol_matrix.index.tolist()
```


```python
df.columns
```




    Index(['contractID', 'symbol', 'expiration', 'strike', 'type', 'last', 'mark',
           'bid', 'bid_size', 'ask', 'ask_size', 'volume', 'open_interest', 'date',
           'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
           'days_to_maturity', 'spot_price', 'moneyness'],
          dtype='object')




```python
cols_to_map = [
        'contractID', 'symbol', 'expiration', 'type', 'last', 'mark',
        'bid', 'bid_size', 'ask', 'ask_size', 'volume', 'open_interest', 'date',
        'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
        'spot_price', 'moneyness'
]
for col in cols_to_map:
    for i,row in hottest_contracts.iterrows():
        hottest_contracts.at[i,col] = indexed.loc[(row['k'],row['t']),col]
```


```python
hottest_contracts.columns
```




    Index(['t', 'k', 'volume', 'contractID', 'symbol', 'expiration', 'type',
           'last', 'mark', 'bid', 'bid_size', 'ask', 'ask_size', 'open_interest',
           'date', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho',
           'spot_price', 'moneyness'],
          dtype='object')



# Plots


```python
"""
volumes scatter
"""
scatter_varbs  = [
   'last', 'mark', 'bid', 'bid_size', 'ask', 'ask_size', 'open_interest', 'volume' 
   ]
df[['k','t']] = df[['strike','days_to_maturity']]

for varb in scatter_varbs:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = df['k'].values
    ys = df['t'].values
    zs = df[varb].values
    ax.scatter(xs, ys, zs, marker='^', color='black')
    ax.set_xlabel('strike')
    ax.set_ylabel('maturity')
    plt.title(varb)
    plt.show()
    plt.clf()
```


    
![png](output_14_0.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_14_2.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_14_4.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_14_6.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_14_8.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_14_10.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_14_12.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_14_14.png)
    



    <Figure size 640x480 with 0 Axes>



```python
"""
volumes scatter
"""
scatter_varbs = [
    ('^','volume','orange'),
    ('*','open_interest','green'),
]

for m, varb, c in scatter_varbs:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = hottest_contracts['k'].values
    ys = hottest_contracts['t'].values
    zs = hottest_contracts[varb].values
    ax.scatter(xs, ys, zs, marker=m, color=c)
    ax.set_xlabel('strike')
    ax.set_ylabel('maturity')
    plt.title(varb)
    plt.show()
    plt.clf()
```


    
![png](output_15_0.png)
    



    <Figure size 640x480 with 0 Axes>



    
![png](output_15_2.png)
    



    <Figure size 640x480 with 0 Axes>



```python
"""
volatility surface
"""
KK,TT = np.meshgrid(K,T)
VV = np.array(
        [[vol_matrix.loc[k,t] for k in K] for t in T]
        )
fig = plt.figure(figsize=plt.figaspect(0.5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(KK, TT, VV, rstride=1, cstride=1, cmap=cm.magma, linewidth=0.1)


ax1.set_zlim(0, float(max(vol_matrix.values.flatten())))
ax1.set_title('volatiltiy surface')
ax1.set_xlabel('strike')
ax1.set_ylabel('maturity')

# Second subplot for wireframe plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_wireframe(KK, TT, VV, rstride=1, cstride=1,color='black')
ax2.set_zlim(0, float(max(vol_matrix.values.flatten())))
ax2.set_title('volatiltiy wireframe')
ax2.set_xlabel('strike')
ax2.set_ylabel('maturity')

plt.tight_layout()
plt.show()
```


    
![png](output_16_0.png)
    

