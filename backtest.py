import yfinance as yf
import pandas as pd
import talib
from config import *

def fetch_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", interval='1h')
    df.dropna(inplace=True)
    return df

def compute_indicators(df):
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['SMA'] = talib.SMA(df['Close'], timeperiod=20)
    return df

def breakout_strategy(df):
    position = 0
    buy_price = 0
    trades = []

    for i in range(1, len(df)):
        if position == 0 and df['High'][i] > df['High'][i-1] * breakout_threshold:
            position = trade_size
            buy_price = df['Close'][i]
            trades.append(('buy', df.index[i], buy_price))
        elif position > 0:
            if df['Close'][i] >= buy_price * take_profit:
                trades.append(('sell_tp', df.index[i], df['Close'][i]))
                position = 0
            elif df['Close'][i] <= buy_price * stop_loss:
                trades.append(('sell_sl', df.index[i], df['Close'][i]))
                position = 0
    return trades