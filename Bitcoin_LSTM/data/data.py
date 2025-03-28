import yfinance as yf

data = yf.download('BTC-USD',start='2010-04-20',end='2025-03-20',interval='1d')

data.to_csv('data/bitcoin_price.csv')