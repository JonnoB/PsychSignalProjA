data = pd.csv_read('../data/data2010to2015.csv')
#let's assume format is: date | sym  | cluster
#date is daily
#sym is cashtag/ticker
#cluster is a number that indicates cluster

import Quandl as q

tickers = data['sym'].unique()
for sym in tickers:   
    try:
        tmp=q.get("WIKI/"+sym+".11",start='2010-01-01', end='2016-01-01', transformation='rdiff')
        tmp[sym]=sym
    except:
        tmp= #NaNs dataframe
    data=pd.merge(data,tmp,axis=1)

rdata=pd.merge(rdata,data,on=['sy'])
mi = pd.read_csv('../data/sp500.csv')

data = data.set_index(pd.to_datetime(data['Date']))
data.drop('Date', axis=1, inplace=True)

mi = mi.set_index(pd.to_datetime(mi['Date']))
mi.drop('Date', axis=1, inplace=True)
