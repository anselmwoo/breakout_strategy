import pandas as pd
import pandas_ta as ta
import yfinance as yf

def fetch_data(symbol='RCAT', period='90d', interval='15m'):
    df = yf.download(symbol, period=period, interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def compute_indicators(df):
    # 计算短期和长期简单移动平均线
    df['ma_short'] = ta.sma(df['close'], length=20)
    df['ma_long'] = ta.sma(df['close'], length=50)
    # 计算ATR指标用于波动率评估
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

def breakout_strategy(df):
    df = compute_indicators(df)
    df['position'] = 0
    # 当价格突破长期均线时买入
    df.loc[df['close'] > df['ma_long'], 'position'] = 1
    # 当价格跌破短期均线时卖出（空仓）
    df.loc[df['close'] < df['ma_short'], 'position'] = -1
    # 计算策略每日收益（持仓乘以价格变化率）
    df['strategy_returns'] = df['position'].shift(1) * df['close'].pct_change()
    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()
    return df
