#Implement Monte Carlo to simulate a stock portfolio

import pandas as pd #Data strctures, data manipulation (SQL), data visualization
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
#pip install pandas numpy matplotlib datetime pandas_datareader
#import data
def get_data(stocks, start, end):
    
    #stockData = pdr.get_data_yahoo(stocks, start, end)
    #stockData = stockData['Close']
    stockData = yf.download(stocks, start=start, end=end)['Close']

    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return  meanReturns, covMatrix

stockList = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] 
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

#Create weights for the portfolio
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

#Monte Carlo Simulation
mc_sims = 100 #Number of simulations
T = 100 #Timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)


portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initalPortfolio = 10000

for m in range(0, mc_sims):
    #Monte Carlo 
    #Assume daily return are distrimuted by a Multivariate Normal Distribution
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix) #Only the diagonal and values under have non-zero values
    #Used to transform independent normal random variables into correlated normal random variables
    dailyReturns = meanM.T * np.inner(L, Z) #Transpose meanM for right dimensions
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1)*initalPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()