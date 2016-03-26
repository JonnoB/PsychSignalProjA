# def reg_with_out_CSI(file, factors):
#
#     file_path = c.clusters_folder_without_CSI + file.value
#                 twt = u.get_clustered_data(file_path)
#
#                 stocks_to_check = twt.SYMBOL.unique()
#                 print(stocks_to_check)
#                 stocks_to_check.savetxt(c.reg_results_folder+'Stocks_in_the_network_based_on_'+file.key)
#
#                 mi = u.get_stock_returns(c.market_return_db, c.market_returns_index, c.market_returns_column, c.start, c.end, c.api_token)
#                 results_dict = {}
#
#                 daily_mean_SI = twt.groupby(['TIMESTAMP_UTC', 'ClusterID']).mean()
#                 daily_mean_SI = daily_mean_SI.reset_index()
#                 daily_mean_SI.columns = ['TIMESTAMP_UTC', 'ClusterID', 'MeanSI']
#
#                 for stock in ['AIG']:  # stocks_to_check.values.tolist():  # twt['SYMBOL'].unique():
#                     print(stock)
#                     stock_r = u.get_stock_returns(c.stock_returns_db, stock, c.stock_returns_col, c.start, c.end, c.api_token)
#                     print(stock_r)
#                     if not stock_r.empty:
#                         print('Downloaded stock data for ' + stock)
#
#                         # Resample stock returns
#                         stock_r = stock_r.resample(period, how=u.cumret, label='right')
#                         print(stock_r)
#                         print(len(stock_r))
#                         stock_r = stock_r.loc[stock_r.index > actualstart]
#                         print(stock_r)
#                         print(len(stock_r))
#
#                         print(stock_r.head())
#
#                         # Detrend data
#                         r_degree = u.integration_degree(stock_r[returns_column_name], max_degree, df_reg_type, alpha)
#                         stock_r = u.detrend(r_degree, stock_r)
#
#                         # Resample & detrend market returns
#                         custom_mi = mi.resample(c.period, how=u.cumret, label='right')
#                         custom_mi = custom_mi.loc[custom_mi.index > c.actual_start]
#                         # print(custom_mi.head())
#                         custom_mi = u.detrend(r_degree, custom_mi)  # this alligns market timeseries to stock returns
#
#                         # Create CSI for a stock
#                         mask = (twt['SYMBOL'] == stock) & (twt['TIMESTAMP_UTC'] >= pd.to_datetime(start)) & (
#                         twt['TIMESTAMP_UTC'] < pd.to_datetime(end))
#                         stock_daily_clusters = twt[mask]
#                         stock_si = pd.merge(stock_daily_clusters, daily_mean_SI, on=['TIMESTAMP_UTC', 'ClusterID'], how='left')
#                         stock_si = stock_si[['TIMESTAMP_UTC', 'MeanSI']]
#                         stock_si.set_index('TIMESTAMP_UTC', inplace=True)
#                         stock_si = stock_si.resample(period, how='mean', label='right')
#
#                         market_model_reg = u.market_model(stock_r, custom_mi)
#                         sentiment_model_reg = u.sentiment_model(market_model_reg, stock_r, stock_si)
#
#                         results_dict[stock] = {'degree of integration': r_degree,
#                                                'returns_ts': stock_r,
#                                                'market_ts': custom_mi,
#                                                'sentiment_ts': stock_si,
#                                                'market_model': market_model_reg,
#                                                'sentiment_model': sentiment_model_reg
#                                                }
#                         print(results_dict[stock]['sentiment_model'].summary())
#                         finaleval.append([stock, results_dict[stock]['sentiment_model'].pvalues[2]])
#
#                     else:
#                         print('Returns of stock ' + stock + ' were not found in the ' + stock_returns_db + ' database')
