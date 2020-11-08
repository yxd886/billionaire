import json
import numpy as np
import os
import pandas as pd
import requests

url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1588262400&end=9999999999&period=900'
response = requests.get(url)
d = response.json()
print(d)

df = pd.DataFrame(d)
original_columns=[u'date',u'close', u'high', u'low', u'open',u'volume',u'weightedAverage']
new_columns = ['Timestamp','Close','High','Low','Open','Volume',"WeightedAverage"]
df = df.loc[:,original_columns]
df.columns = new_columns
print(df.head())

df.to_csv('data/eth.csv',index=None)