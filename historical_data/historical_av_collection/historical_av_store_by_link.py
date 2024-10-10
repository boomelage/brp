# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:02:17 2024

@author: boomelage
"""

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
import pandas as pd
from tqdm import tqdm

from model_settings import ms
import requests
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import QuantLib as ql
from datetime import datetime
import time
from historical_alphaVantage_collection import collect_av_link

key = ms.av_key
symbol = 'SPY'
date = '2024-10-10'


underlying_url = str(
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
    f"&symbol={symbol}&date={date}&outputsize=full&apikey={key}"
    )
spotr = requests.get(underlying_url)
spots = pd.DataFrame(spotr.json()['Time Series (Daily)']).T
spots = spots.astype(float)['4. close']

dates = spots.index.tolist()[:1000]



for date in dates:
    spot = float(spots[date])
    link = collect_av_link(date, symbol, spot, key)
    printdate = datetime.strptime(date, '%Y-%m-%d').strftime('%A, %Y-%m-%d')
    while True:
        try:
            with pd.HDFStore(r'alphaVantage vanillas.h5') as store:
                hdf_datekey = str('date_' + date.replace('-', '_'))
                
                store.append(
                    f"{hdf_datekey}/raw_data", link['raw_data'],
                    format='table', append=True
                )
                store.append(
                    f"{hdf_datekey}/surface", link['surface'],
                    format='table', append=True
                )
                store.append(
                    f"{hdf_datekey}/hottest_contracts", 
                    link['hottest_contracts'],
                    format='table', append=True
                )
                print(f"collected {printdate}")
            break

        except Exception as OSError:
            print(f"error for {printdate}:\n{OSError}")
            time.sleep(5)