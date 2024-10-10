from model_settings import ms
import requests
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import QuantLib as ql
from datetime import datetime


def generate_daily_dates(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strings = date_range.strftime('%Y-%m-%d').tolist()
    return date_strings


key = ms.av_key
symbol = 'SPY'
end_date = '2024-05-17'
dates = generate_daily_dates('2024-05-17', end_date)

underlying_url = str(
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
    f"&symbol={symbol}&date={end_date}&outputsize=full&apikey={key}"
    )
spotr = requests.get(underlying_url)
spots = pd.DataFrame(spotr.json()['Time Series (Daily)']).T
spots = spots.astype(float)

spots.index = pd.to_datetime(spots.index,format='%Y-%m-%d')



for date in dates:
    options_url = str(
        "https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&"
        f"symbol={symbol}"
        f"&date={date}"
        f"&apikey={key}"
              )
    spot = float(spots['4. close'][date])

    r = requests.get(options_url)
    data = r.json()
    
    
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
    
    indexed = df.copy().set_index(['strike','days_to_maturity'])
    
    T = np.sort(df['days_to_maturity'].unique()).tolist()
    K = np.sort(df['strike'].unique()).tolist()
    volume_heatmap = pd.DataFrame(
        np.full((len(K), len(T)), np.nan), index=K, columns=T)
    for k in K:
        for t in T:
            try:
                volume_heatmap.loc[k,t] = indexed.loc[(k,t),'volume']
            except Exception:
                pass
    
    hottest_contracts = pd.DataFrame(volume_heatmap.unstack().sort_values(
        ascending=False)).head(50).reset_index()
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
                vol_matrix.loc[k,t] = indexed.loc[
                    (k,float(t)),'implied_volatility']
            except Exception:
                pass
    
    vol_matrix = vol_matrix.dropna().copy()
    T = vol_matrix.columns.tolist()
    K = vol_matrix.index.tolist()
    
    print(vol_matrix)
    # KK,TT = np.meshgrid(K,T)
    # VV = np.array(
    #         [[vol_matrix.loc[k,t] for k in K] for t in T]
    #         )
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # surf = ax1.plot_surface(KK, TT, VV, rstride=1, cstride=1, cmap=cm.magma, linewidth=0.1)
    
    # ax1.set_zlim(0, float(max(vol_matrix.values.flatten())))
    # ax1.set_title('volatiltiy surface')
    # ax1.set_xlabel('strike')
    # ax1.set_ylabel('maturity')
    
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax2.plot_wireframe(KK, TT, VV, rstride=1, cstride=1,color='black')
    # ax2.set_zlim(0, float(max(vol_matrix.values.flatten())))
    # ax2.set_title('volatiltiy wireframe')
    # ax2.set_xlabel('strike')
    # ax2.set_ylabel('maturity')
    
    # plt.tight_layout()
    # plt.show()
    
