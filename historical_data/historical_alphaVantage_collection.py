# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:28:49 2024

@author: boomelage
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures
pd.set_option("display.max_columns",None)

def generate_daily_dates(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_strings = date_range.strftime('%Y-%m-%d').tolist()
    return date_strings


def collect_option_chain_link(date, symbol, key):
    option_chain_link = {}
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
    
    calls = df[df['w']=='call'].copy().reset_index(drop=True)
    T = np.sort(calls['days_to_maturity'].unique().astype(float)).tolist()
    K = np.sort(calls['strike_price'].unique().astype(float)).tolist()
    calls = calls.set_index(['days_to_maturity','strike_price'])
    call_surf = pd.DataFrame(
        np.zeros((len(K),len(T)),dtype=float),
        index = K,
        columns = T
        )
    for k in K:
        for t in T:
            try:
                call_surf.loc[k,t] = calls.loc[(t,k),'volatility']
            except Exception:
                call_surf.loc[k,t] = np.nan
                
    
    puts = df[df['w']=='put'].copy().reset_index(drop=True)
    T = np.sort(puts['days_to_maturity'].unique().astype(float)).tolist()
    K = np.sort(puts['strike_price'].unique().astype(float)).tolist()
    puts = puts.set_index(['days_to_maturity','strike_price'])
    put_surf = pd.DataFrame(
        np.zeros((len(K),len(T)),dtype=float),
        index = K,
        columns = T
        )
    for k in K:
        for t in T:
            try:
                put_surf.loc[k,t] = puts.loc[(t,k),'volatility']
            except Exception:
                put_surf.loc[k,t] = np.nan
    
    option_chain_link['calls'] = {
        'surface':call_surf,
        'contracts': calls
        }
    
    option_chain_link['puts'] = {
        'surface':put_surf,
        'contracts':puts
        }
    
    return option_chain_link

def collect_chain(start_date, end_date, symbol, key):
    dates = generate_daily_dates(start_date, end_date)
    chain = {}
    def fetch_data(date):
        try:
            return date, collect_option_chain_link(date, symbol, key)
        except Exception:
            calculation_datetime = datetime.strptime(
                date, "%Y-%m-%d").strftime("%A, %Y-%m-%d")
            print(f"no data for: {calculation_datetime}")
            return date, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_data, dates))
    chain = {date: data for date, data in results if data is not None}
    return chain

from model_settings import ms
key = ms.av_key
symbol = 'SPY'
start_date = '2008-01-01'
end_date = datetime.today().strftime("%Y-%m-%d")
chain = collect_chain(start_date, end_date, symbol, key)