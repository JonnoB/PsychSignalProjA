__author__ = 'cparlin'
__author__ = 'stan'

import utility as u
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt

bmth_us = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())

finaleval = []

file='../data/BULL_BEAR_MSG_RATIO.csv'
twt=u.get_clustered_data(file)
col_names=u.get_raw_data_col_names()

stock_returns_db="WIKI"
market_return_db="YAHOO"
market_returns_index='INDEX_GSPC'
market_returns_column='6'
start='2010-10-01'
actualstart = '2011-01-01'
end='2016-01-01'
period='2M'
api_token= 'c54mBskiz_BsF4vWWL2s'
max_degree=3
returns_column_name="Adj. Close"
df_reg_type='c'
alpha=0.05

stocks_to_check = pd.read_csv("../data/bear_bull_ratio_stocknames.csv", header=None)

print(stocks_to_check)

mi = u.get_stock_returns(market_return_db, market_returns_index, market_returns_column, start, end, api_token)

results_dict = {}
stock_returns_col='11'

daily_mean_SI=twt.groupby(['TIMESTAMP_UTC', 'ClusterID']).mean()
daily_mean_SI=daily_mean_SI.reset_index()
daily_mean_SI.columns=['TIMESTAMP_UTC', 'ClusterID', 'MeanSI']


for stock in stocks_to_check.values.ravel().tolist(): # twt['SYMBOL'].unique():
    print(stock)
    stock_r = u.get_stock_returns(stock_returns_db, stock, stock_returns_col, start, end, api_token)

    if not stock_r.empty:
        print('Downloaded stock data for ' + stock)
        stock_r = stock_r.resample(period, how=u.volatility,label='right')
        stock_r = stock_r.loc[stock_r.index > actualstart]
        stock_r = u.detrend(1, stock_r)
        custom_mi = mi.resample(period, how=u.cumret,label='right')
        custom_mi = custom_mi.loc[custom_mi.index > actualstart]
        custom_mi = u.detrend(1, custom_mi) #this alligns market timeseries to stock returns

        mask = (twt['SYMBOL'] == stock) #& (twt['TIMESTAMP_UTC']>=pd.to_datetime(start)) & (twt['TIMESTAMP_UTC']<pd.to_datetime(end))
        stock_si = twt[mask]
        stock_si = stock_si[['TIMESTAMP_UTC', 'mean']]
        stock_si.set_index('TIMESTAMP_UTC', inplace=True)
        stock_si = u.detrend(1, stock_si)

        market_model_reg = u.market_model(stock_r, custom_mi)
        sentiment_model_reg = u.sentiment_model(market_model_reg, stock_r, stock_si)

        results_dict[stock]={'degree of integration':1,
                             'returns_ts':stock_r,
                             'market_ts':custom_mi,
                             'sentiment_ts':stock_si,
                             'market_model':market_model_reg,
                             'sentiment_model':sentiment_model_reg
                             }
        print(results_dict[stock]['sentiment_model'].summary())
        finaleval.append([stock, results_dict[stock]['sentiment_model'].params[3], results_dict[stock]['sentiment_model'].pvalues[3]])

    else:
        print('Returns of stock ' + stock + ' were not found in the ' + stock_returns_db + ' database')

print(finaleval)
pd.DataFrame(finaleval).to_csv('../output/evaluation_pvalues.csv')
