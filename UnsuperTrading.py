from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
#import pandas_ta as ta
import warnings

# pip install statsmodels pandas-datareader matplotlib pandas numpy datetime yfinance pandas_ta

# Suppress warnings
warnings.filterwarnings('ignore')

#######################################################################################
# Set up dataframe with S&P500 stocks
#######################################################################################

SP500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

SP500['Symbol']= SP500['Symbol'].str.replace('.','-')
symbols_list = SP500['Symbol'].unique().tolist()

#print(symbols_list)

end_date = '2024-12-01'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8) #Eight years of time

df = yf.download(tickers = symbols_list, 
                 start = start_date,
                 end = end_date).stack()
#Stack makes sure each row is unique, pivots columns into a multi-level index

df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower() #Covert all column names to lower case

#print(df)

#######################################################################################
# Calculate features and technical indicators for each stock
#######################################################################################

# Garman-Klass volatility, estimates volatility using FOUR key metrics
    #High: Highest price of asset during period
    #Low: Lowest price of asset during peirod
    #Close: Closing price of asset
    #Open: Opening price of asset
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2 - ((2*np.log(2)-1)*(np.log(df['adj close'])-np.log(df['open']))**2)
print(df)

#Relative Strength Index (RSI), used to measure speed and change of price movements of a stock from 0 to 100

#df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: ta.rsi(close=x, length=20))



