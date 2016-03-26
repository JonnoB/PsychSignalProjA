__author__ = 'cparlin'
__author__ = 'stan'

import cutility as u
import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
from config import config as c

#bmth_us = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())

csi_clusters_file_dict = c.csi_clusters_file_dict


def reg_clusters_with_CSI(file):
    file_path = c.clusters_folder_with_CSI + file
    twt = u.get_cluster_data_with_CSI(file_path)
    print(twt.head())

    stocks_to_check = twt['SYMBOL'].unique()
    print(stocks_to_check)
    # np.savetxt(c.reg_results_folder + 'Stocks_in_the_network_based_on_' + file, stocks_to_check)

    mi = u.get_stock_returns(c.market_return_db, c.market_returns_index, c.market_returns_column, c.start, c.end,
                             c.api_token)

    custom_mi = mi.resample(c.period, how=u.cumret, label='right')
    custom_mi = custom_mi.loc[custom_mi.index > c.actual_start]
    print(custom_mi)

    results_dict = {}

    for stock in ['IBM']:  # stocks_to_check.values.tolist():
        print(stock)
        stock_returns = u.get_stock_returns(c.stock_returns_db, stock, c.stock_returns_col, c.start, c.end, c.api_token)
        result_df = pd.DataFrame(columns=['Mean sentiment model on returns',
                                          #'Percentage change in sentiment model on returns',
                                          'Mean sentiment model on volatility',
                                          #'Percentage change in sentiment model on volatility'
                                 ]
                                 )
        if not stock_returns.empty:
            print('Downloaded stock data for ' + stock)
            stats_to_save = []

            # Resample stock returns
            stock_r = stock_returns.resample(c.period, how=u.cumret, label='right')
            stock_v = stock_returns.resample(c.period, how=u.volatility, label='right')
            # in current setup this creates 30 bi-monthly periods from 01-2011-->02-2011 to 10-2015-->12-2015:
            stock_r = stock_r.loc[stock_r.index > c.actual_start]
            stock_v = stock_v.loc[stock_v.index > c.actual_start]

            # Name timeseries
            mask = (twt['SYMBOL'] == stock)
            stock_period_clusters = twt[mask]

            stock_sentiment_ts = stock_period_clusters['mean']
            stock_r_ts = stock_r[c.returns_column_name]
            stock_v_ts = stock_v[c.returns_column_name]
            market_index_ts = custom_mi['Adjusted Close']


            # Market model on returns
            mm_list_of_factors = [stock_r_ts, market_index_ts]
            market_model_reg, mm_degree = u.run_model(mm_list_of_factors, u.market_model)

            sm_list_of_factors = [market_model_reg.resid, u.t_plus_n_minus_1(stock_r_ts, 0), u.t_plus_n_minus_1(stock_sentiment_ts, 0)]
            print([len(x) for x in sm_list_of_factors])

            sentiment_model_reg, sm_degree = u.run_model(sm_list_of_factors, u.sentiment_model)

            # per_change_list_of_factors = u.change_in_sentiment_model_factor_list(market_model_reg.resid, stock_r_ts,
            #                                                                      stock_sentiment_ts)
            #
            # per_change_model, pc_degree = u.run_model(per_change_list_of_factors, u.sentiment_model)

            models_dict = {'Market_model_returns': [market_model_reg, mm_degree],
                           'Mean_Sentiment_model_returns': [sentiment_model_reg, sm_degree],
                           #'Percentage_change_in_sentiment_model_returns': [per_change_model, pc_degree]
                           }

            # save summary stats to a list
            cell = u.format_result(sentiment_model_reg.params[2], sentiment_model_reg.pvalues[2])
            stats_to_save.append(cell)
            # cell = u.format_result(per_change_model.param[2], per_change_model.pvalue[2])
            # stats_to_save.append(cell)


            # Market model on volatility - rerun but on volatility

            mm_list_of_factors = [stock_v_ts, market_index_ts]
            market_model_reg, mm_degree = u.run_model(mm_list_of_factors, u.market_model)

            sm_list_of_factors = [market_model_reg.resid, u.t_plus_n_minus_1(stock_r_ts, mm_degree), u.t_plus_n_minus_1(stock_sentiment_ts, mm_degree)]
            sentiment_model_reg, sm_degree = u.run_model(sm_list_of_factors, u.sentiment_model)

            # per_change_list_of_factors = u.change_in_sentiment_model_factor_list(market_model_reg.resid, stock_v_ts,
            #                                                                      stock_sentiment_ts)
            # per_change_model, pc_degree = u.run_model(per_change_list_of_factors, u.sentiment_model)

            # save summary stats to a list
            cell = u.format_result(sentiment_model_reg.params[2], sentiment_model_reg.pvalues[2])
            stats_to_save.append(cell)
            # cell = u.format_result(per_change_model.param[2], per_change_model.pvalue[2])
            # stats_to_save.append(cell)

            models_dict = {**{'Market_model_volatility': [market_model_reg, mm_degree],
                           'Mean_Sentiment_model_volatility': [sentiment_model_reg, sm_degree],
                          # 'Percentage_change_in_sentiment_model_volatility': [per_change_model, pc_degree]
                              },
                           **models_dict}

            results_dict[stock] = {'returns_ts': stock_r_ts,
                                   'volatility_ts': stock_v_ts,
                                   'market_ts': market_index_ts,
                                   'sentiment_ts': stock_sentiment_ts,
                                    'dict_of_models': models_dict}

            df = pd.DataFrame(stats_to_save, columns=result_df.columns)
            result_df = result_df.append(df)

        else:
            print('Returns of stock ' + stock + ' were not found in the ' + c.stock_returns_db + ' database')

    return result_df, results_dict

for key, file in {'Bearish Intensity': 'BEARISH_INTENSITY.csv'}.items(): #csi_clusters_file_dict.items():
    df_result, dict_of_results = reg_clusters_with_CSI(file)
    df_result.to_csv(c.reg_results_folder+key+'_results.csv')

print("Done")
