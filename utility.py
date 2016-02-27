__author__ = 'stan'

import pandas as pd
import numpy as np
import os
import Quandl as q
import statsmodels.tsa.stattools as stat
import statsmodels.api as sm


def get_clustered_data(file_name):
    path = (os.getcwd() + '\\')
    df = pd.read_csv(filepath_or_buffer=path + file_name, delimiter=",", header=0)
    df = df.iloc[:, 1:]
    df['TIMESTAMP_UTC'] = pd.to_datetime(arg=df.TIMESTAMP_UTC, infer_datetime_format='true', unit='D')
    return df


def get_raw_data_col_names():
    return ['sym', 'date', 'sen', 'cluster']


def get_stock_returns(db, sym, column, start, end, api_token):
    try:
        rs = q.get(dataset=(db + '/' + sym + "." + column), trim_start=start, trim_end=end, transformation='rdiff', \
                   authtoken=api_token)
        return rs
    except:
        return pd.DataFrame({})


def resample(ts, periods):
    return ts.resample(periods, how=cumret)


def aug_dickey_fuller(ts, reg_type, alpha):
    dft = stat.adfuller(ts, regression=reg_type, autolag="BIC", store=True, regresults=True)
    pvalue = dft[1]
    return pvalue <= alpha


def take_difference(ts):
    ts = t_plus_n(ts, 1) - t_plus_n_minus_1(ts, 0)
    return ts


def integration_degree(ts, max_degree, reg_type, alpha):
    df_passed = False
    iteration = 0
    while (df_passed is False) & (iteration <= max_degree):
        dft = aug_dickey_fuller(ts, reg_type, alpha)
        if dft:
            df_passed = True
        else:
            ts = take_difference(ts)
            iteration += 1
    return iteration


def detrend(degree, ts):
    for i in range(0, degree):
        ts = take_difference(ts)
    return ts


def t_plus_n_minus_1(ts, n):
    return ts[n:(len(ts) - 1)]


def t_plus_n(ts, n):
    return ts[n:len(ts)]


def market_model(stock_ts, market_ts):
    r_t_plus_one = t_plus_n(stock_ts, 1)
    r_t = t_plus_n_minus_1(stock_ts, 0)
    m_t = t_plus_n_minus_1(market_ts, 0)

    dep_vars = np.concatenate((r_t.values,m_t.values), axis=1)
    dep_vars = sm.add_constant(dep_vars)
    linreg = sm.OLS(r_t_plus_one.values, dep_vars)

    return linreg.fit()


def sentiment_model(market_model, stock_ts, sentiment_ts):
    y = market_model.resid
    r_t = t_plus_n_minus_1(stock_ts, 0)
    si_t = t_plus_n_minus_1(sentiment_ts, 0)

    dep_vars = np.concatenate((r_t.values, si_t.values), axis=1)
    dep_vars = sm.add_constant(dep_vars)
    linreg = sm.OLS(y, dep_vars)

    return linreg.fit()


def mean_si(df):
    gdf = df.groupby('TIMESTAMP_UTC', 'ClusterID').mean()


def cumret(ts):
    ts = np.add(ts, 1)
    ts = np.cumprod(ts)[len(ts) - 1]
    ts = ts - 1
    return ts
