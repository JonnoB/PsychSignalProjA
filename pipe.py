__author__ = 'stan'

import utility as u
import pandas as pd
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm

bmth_us = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())

file = 'Twitaggmat.csv'
twt = u.get_clustered_data(file)
col_names = u.get_raw_data_col_names()

stock_returns_db = "WIKI"
market_return_db = "YAHOO"
market_returns_index = 'INDEX_GSPC'
market_returns_column = '6'
start = '2011-01-01'
end = '2015-01-01'
period = '2M'
api_token = 'c54mBskiz_BsF4vWWL2s'
max_degree = 3
returns_column_name = "Adj. Close"
df_reg_type = 'c'
alpha = 0.05

mi = u.get_stock_returns(market_return_db, market_returns_index, market_returns_column, start, end, api_token)

results_dict = {}
stock_returns_col = '11'

daily_mean_SI = twt.groupby(['TIMESTAMP_UTC', 'ClusterID']).mean()
daily_mean_SI = daily_mean_SI.reset_index()
daily_mean_SI.columns = ['TIMESTAMP_UTC', 'ClusterID', 'MeanSI']

for stock in ['AIG', 'AMD']:  # twt['SYMBOL'].unique():
    stock_r = u.get_stock_returns(stock_returns_db, stock, stock_returns_col, start, end, api_token)

    if not stock_r.empty:
        print('Downloaded stock data for ' + stock)
        stock_r = stock_r.resample(period, how=u.cumret)
        r_degree = u.integration_degree(stock_r[returns_column_name], max_degree, df_reg_type, alpha)
        stock_r = u.detrend(r_degree, stock_r)
        custom_mi = mi.resample(period, how=u.cumret)
        custom_mi = u.detrend(r_degree, custom_mi)  # this alligns market timeseries to stock returns

        mask = (twt['SYMBOL'] == stock) & (twt['TIMESTAMP_UTC'] >= pd.to_datetime(start)) & (
        twt['TIMESTAMP_UTC'] < pd.to_datetime(end))
        stock_daily_clusters = twt[mask]
        stock_si = pd.merge(stock_daily_clusters, daily_mean_SI, on=['TIMESTAMP_UTC', 'ClusterID'], how='left')
        stock_si = stock_si[['TIMESTAMP_UTC', 'MeanSI']]
        stock_si.set_index('TIMESTAMP_UTC', inplace=True)
        stock_si = stock_si.resample(period, how='mean')

        market_model_reg = u.market_model(stock_r, custom_mi)
        sentiment_model_reg = u.sentiment_model(market_model_reg, stock_r, stock_si)

        results_dict[stock] = {'degree of integration': r_degree,
                               'returns_ts': stock_r,
                               'market_ts': custom_mi,
                               'sentiment_ts': stock_si,
                               'market_model': market_model_reg,
                               'sentiment_model': sentiment_model_reg
                               }

    else:
        print('Returns of stock ' + stock + ' were not found in the ' + stock_returns_db + ' database')

print(results_dict['AMD']['sentiment_model'].summary())
print(results_dict['AIG']['sentiment_model'].summary())
print(results_dict['AIG']['market_ts'])
