__author__ = 'cparlin'

import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm

bmth_us = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())

data = pd.read_csv('../data/apple2010to2015daily.csv')
mi = pd.read_csv('../data/sp500.csv')

data = data.set_index(pd.to_datetime(data['Date']))
data.drop('Date', axis=1, inplace=True)

mi = mi.set_index(pd.to_datetime(mi['Date']))
mi.drop('Date', axis=1, inplace=True)

print(mi.head())

def cumret(ts):
    ts=np.add(ts,1)
    ts=np.cumprod(ts)[len(ts)-1]
    ts = ts-1
    return ts

data = data.resample(bmth_us, how=cumret)
mi = mi.resample(bmth_us, how=cumret)

data_stattest = ts.adfuller(data['Adj. Close'], regression='c', autolag="BIC", store=True, regresults=True)
mi_stattest = ts.adfuller(mi['Adj Close'], regression='c', autolag="BIC", store=True, regresults=True)

print("Stock Stationarity Test:")
print(data_stattest)

print("Market Index Stationarity Test:")
print(mi_stattest)

alldata = np.concatenate((data, mi), axis=1)

alldata = sm.add_constant(alldata)

print(alldata[0:5])

print(alldata[1:len(alldata),1].shape)
print(alldata[0:(len(alldata)-1), [0,2]].shape)

linreg = sm.OLS(alldata[1:len(alldata),1], alldata[0:(len(alldata)-1), [0,2]])
results = linreg.fit()
print(results.summary())

residuals = results.resid

res_stattest = ts.adfuller(residuals, regression='c', autolag="BIC", store=True, regresults=True)

print("Residuals Stationarity Test:")
print(res_stattest)