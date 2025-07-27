import pandas as pd
import yfinance as yf
import ta

def fetch_data(symbol='RCAT', period='90d', interval='15m'):
    df = yf.download(symbol, period=period, interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def compute_indicators(df):
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    # EMA
    df['ema_short'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(close=df['close'], window=50).ema_indicator()
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    return df

def breakout_strategy(df):
    df = compute_indicators(df)
    df['position'] = 0
    df.loc[df['close'] > df['ema_long'], 'position'] = 1
    df.loc[df['close'] < df['ema_short'], 'position'] = -1
    df['strategy_returns'] = df['position'].shift(1) * df['close'].pct_change()
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    return df
