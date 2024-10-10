from pathlib import Path
from model_settings import ms
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import QuantLib as ql
import time

current_dir = os.path.abspath(str(Path()))

store = pd.HDFStore(r'alphaVantage Vanillas.h5')
keys = store.keys()

print(len(keys))
store.close()

contract_keys = [key for key in keys if key.find('hottest_contracts')!= -1]
print(len(contract_keys))

surface_keys = [key for key in keys if key.find('surface')!= -1]
print(len(surface_keys))

raw_data_keys = [key for key in keys if key.find('raw_data')!=-1]
print(len(raw_data_keys))

"""
loop start
"""
calibration_errors = pd.Series()

for surface_key in surface_keys:
	second_backslash_pos = surface_key.find('/', 1)
	date_key_component = surface_key[:second_backslash_pos+1]
	print(date_key_component)
	accomp = [key for key in contract_keys if f'{date_key_component}hottest_contracts' in contract_keys][0]
	print(accomp)

	raw_data_key = [key for key in raw_data_keys if f'{date_key_component}raw_data' in raw_data_keys][0]
	print(raw_data_key)

	date_string = date_key_component[date_key_component.find('_',0)+1:date_key_component.find('_',0)+11]
	print(date_string)

	calculation_datetime = datetime.strptime(date_string,'%Y_%m_%d')
	print(calculation_datetime)
	while True:
		try:
			store.open()
			raw_data = pd.DataFrame(store[raw_data_key])
			vol_matrix = pd.DataFrame(store[surface_key])
			contracts = pd.DataFrame(store[accomp])
			break
		except OSError:
			print('waiting for pending file operations'
				  '\nretrying in:')
			for i in range(0,5):
				print(5-i)
				time.sleep(1)
		finally:
			store.close()



	pd.to_numeric(raw_data['last'])
	raw_data['date'] = pd.to_datetime(raw_data['date'])
	raw_data['expiration'] = pd.to_datetime(raw_data['expiration'])
	raw_data['implied_volatility'] = pd.to_numeric(raw_data['implied_volatility'])
	raw_data['strike'] = pd.to_numeric(raw_data['strike'])
	raw_data['last'] = pd.to_numeric(raw_data['last'])

	contract_maturities = np.array((raw_data['expiration'] - raw_data['date']) / timedelta(days=1)).astype(int)
	raw_data['days_to_maturity'] = contract_maturities
	raw_data.iloc[:5,:]

	T = vol_matrix.columns.tolist()
	K = vol_matrix.index.tolist()
	spot = float(contracts['spot_price'].unique()[0])
	calculation_date = ql.Date(
	    calculation_datetime.day,
	    calculation_datetime.month,
	    calculation_datetime.year)
	s = spot

	r = 0.04
	g = 0.018


	# pd.set_option("display.max_columns",None)
	# print(f"\n{raw_data.iloc[:5,:]}\n{raw_data.dtypes}\n{vol_matrix}\n"
	# 	  f"\n{contracts}\n{spot} {contracts['spot_price'].unique()}\n")
	# print(f"\nmaturitites:\n     {T}\nstrikes:\n     {K}\n")
	# print(calculation_date)
	# pd.reset_option("display.max_columns")

	ql.Settings.instance().evaluationDate = calculation_date
	flat_ts, dividend_ts = ms.ql_ts_rg(r, g, calculation_date)
	S_handle = ql.QuoteHandle(ql.SimpleQuote(s))

	heston_helpers = []
	v0 = 0.01; kappa = 0.2; theta = 0.02; rho = -0.75; eta = 0.5;
	process = ql.HestonProcess(
	    flat_ts,
	    dividend_ts,
	    S_handle,
	    v0,                # Initial volatility
	    kappa,             # Mean reversion speed
	    theta,             # Long-run variance (volatility squared)
	    eta,               # Volatility of the volatility
	    rho                # Correlation between asset and volatility
	)
	model = ql.HestonModel(process)
	engine = ql.AnalyticHestonEngine(model)

	for t in T:
	    for k in K:
	        p = ql.Period(int(t),ql.Days)
	        volatility = vol_matrix.loc[k,t]
	        helper = ql.HestonModelHelper(
	            p, ms.calendar, float(s), k, 
	            ql.QuoteHandle(ql.SimpleQuote(volatility)), 
	            flat_ts, 
	            dividend_ts
	            )
	        helper.setPricingEngine(engine)
	        heston_helpers.append(helper)

	lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)


	model.calibrate(heston_helpers, lm,
	                ql.EndCriteria(1000, 50, 1.0e-8,1.0e-8, 1.0e-8))

	theta, kappa, eta, rho, v0 = model.params()
	heston_parameters = pd.Series(
	    [theta, kappa, eta, rho, v0],
	    index = ['theta', 'kappa', 'eta', 'rho', 'v0'],
	    dtype = float
	)
	print(heston_parameters)


	calibration_test_data = raw_data.copy()[['strike','type','last','implied_volatility','days_to_maturity']]
	calibration_test_data.columns = ['strike_price','w','market_price','volatility','days_to_maturity']
	calibration_test_data['spot_price'] = s
	calibration_test_data['risk_free_rate'] = r
	calibration_test_data['dividend_rate'] = g
	calibration_test_data = calibration_test_data[calibration_test_data['days_to_maturity'].isin(contracts['days_to_maturity'])]
	calibration_test_data = calibration_test_data[calibration_test_data['days_to_maturity'].isin(contracts['days_to_maturity'])]

	calibration_test_data[heston_parameters.index.tolist()] = np.tile(
		heston_parameters,(calibration_test_data.shape[0],1))
	calibration_test_data.loc[:,'moneyness'] = ms.vmoneyness(
		calibration_test_data['spot_price'].values,
		calibration_test_data['strike_price'].values,
		calibration_test_data['w'].values
		)
	calibration_test_data['calculation_date'] = calculation_datetime
	calibration_test_data['black_scholes'] = ms.vector_black_scholes(
        calibration_test_data)
	calibration_test_data['heston_price'] = ms.vector_heston_price(calibration_test_data)
	calibration_test_data.loc[:,'absolute_error'] = calibration_test_data['heston_price'].values - calibration_test_data['black_scholes'].values

	large_errors = calibration_test_data.copy(
        )[calibration_test_data['absolute_error']>=1]
	calibration_errors[calculation_datetime] = np.mean(
        np.abs(calibration_test_data['absolute_error']))
	print(large_errors.describe())


plt.figure()
plt.plot(calibration_errors,color='black')
plt.ylabel('absolute pricing error')
plt.xticks(rotation=45)
plt.show()
plt.clf()