# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:28:49 2024

@author: boomelage
"""

import requests
import pandas as pd
from model_settings import ms
import numpy as np
pd.set_option("display.max_columns",None)

def generate_daily_dates(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strings = date_range.strftime('%Y-%m-%d').tolist()
    return date_strings



option_chain = {}
def collect_option_chain(start_date, end_date, symbol, key):
    dates = generate_daily_dates(start_date, end_date)
    for date in dates:
        try:
            underlying_url = str(
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
                f"&symbol={symbol}&date={date}&outputsize=full&apikey={key}"
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
            option_chain[date] = (contracts,ivol_df)
        except Exception as e:
            print(e)
    return option_chain


key = ms.av_key
symbol = 'SPY'
start_date = '2024-01-03'
end_date = '2024-01-03'
option_chain = collect_option_chain(start_date, end_date, symbol, key)