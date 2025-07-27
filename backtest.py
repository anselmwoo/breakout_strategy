import yfinance as yf
import pandas as pd
import ta

def fetch_data(symbol='RCAT',
               periods=['7d', '14d', '30d'],
               intervals=['5m', '15m', '1h'],
               min_length=30):
    """
    自动尝试不同周期和间隔，保证至少 min_length 行数据。
    优先从较短周期和细间隔开始，逐步降级。
    """
    for period in periods:
        for interval in intervals:
            df = yf.download(symbol, period=period, interval=interval)
            if df is not None and len(df) >= min_length:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                print(f"成功获取数据：period={period}, interval={interval}, 行数={len(df)}")
                return df
            else:
                print(f"数据不足，尝试下一组合：period={period}, interval={interval}, 行数={len(df) if df is not None else 0}")
    raise ValueError(f"无法获取足够数据 (至少 {min_length} 行) 用于策略回测，建议更换股票或数据源")

def compute_indicators(df):
    if len(df) < 30:
        print("数据长度较短，指标计算可能不够稳定")
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['ema_short'] = ta.trend.EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(close=df['close'], window=21).ema_indicator()
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
