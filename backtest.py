import yfinance as yf
import pandas as pd
import ta

def fetch_data(symbol='RCAT',
               periods=['7d', '14d', '30d'],
               intervals=['5m', '15m', '1h'],
               min_length=30):
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

def compute_indicators(df, rsi_window=14, ema_short_window=9, ema_long_window=21):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=rsi_window).rsi()
    df['ema_short'] = ta.trend.EMAIndicator(close=df['close'], window=ema_short_window).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(close=df['close'], window=ema_long_window).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    return df

import numpy as np

def breakout_strategy(df, rsi_window=14, ema_short_window=9, ema_long_window=21, include_short=True):
    df = compute_indicators(df, rsi_window, ema_short_window, ema_long_window)
    df['position'] = 0
    df.loc[df['close'] > df['ema_long'], 'position'] = 1
    df.loc[df['close'] < df['ema_short'], 'position'] = -1

    df['trade_signal'] = df['position'].diff()

    if include_short:
        # 多空双向收益
        df['strategy_returns'] = df['position'].shift(1) * df['close'].pct_change()
    else:
        # 只计算多头收益，空头收益设为0
        df['strategy_returns'] = np.where(
            df['position'].shift(1) == 1,
            df['close'].pct_change(),
            0
        )

    df['equity_curve'] = (1 + df['strategy_returns'].fillna(0)).cumprod()

    # 交易明细部分保持不变
    trades = df[df['trade_signal'] != 0][['position', 'close', 'trade_signal']].copy()
    trades = trades.rename(columns={'position': 'position_after_trade', 'close': 'trade_price'})
    trades['trade_type'] = trades['trade_signal'].apply(lambda x: 'Buy' if x > 0 else 'Sell')
    trades.index.name = 'datetime'
    trades = trades.reset_index()

    pnl_list = []
    buy_price = None
    for _, row in trades.iterrows():
        if row['trade_type'] == 'Buy':
            buy_price = row['trade_price']
            pnl_list.append(None)
        elif row['trade_type'] == 'Sell' and buy_price is not None:
            pnl = (row['trade_price'] - buy_price) / buy_price
            pnl_list.append(pnl)
            buy_price = None
        else:
            pnl_list.append(None)
    trades['pnl'] = pnl_list

    return df, trades
