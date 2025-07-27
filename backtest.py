import pandas as pd
import numpy as np

def load_data(symbol='AAPL', period='3mo', interval='15m'):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df = df[~df.index.duplicated()]  # 去重
    df = df.between_time("09:30", "16:00")  # 排除夜盘
    return df

def breakout_strategy(df, window=20, stop_loss=0.02, take_profit=0.04):
    df = df.copy()
    df['High_Max'] = df['High'].shift(1).rolling(window).max()
    df['Low_Min'] = df['Low'].shift(1).rolling(window).min()

    position = 0
    entry_price = 0
    signals = []
    profits = []

    for i in range(len(df)):
        if position == 0:
            if df['Close'].iloc[i] > df['High_Max'].iloc[i]:
                position = 1
                entry_price = df['Close'].iloc[i]
                signals.append((df.index[i], 'Buy', entry_price))
        elif position == 1:
            price = df['Close'].iloc[i]
            if price < entry_price * (1 - stop_loss) or price > entry_price * (1 + take_profit):
                exit_price = price
                profit_pct = (exit_price - entry_price) / entry_price
                signals.append((df.index[i], 'Sell', exit_price))
                profits.append((df.index[i], profit_pct))
                position = 0

    df['Signal'] = np.nan
    df['ProfitPct'] = np.nan
    for time, action, price in signals:
        df.loc[time, 'Signal'] = action
    for time, profit in profits:
        df.loc[time, 'ProfitPct'] = profit

    df['StrategyReturn'] = df['ProfitPct'].fillna(0)
    df['CumulativeReturn'] = (1 + df['StrategyReturn']).cumprod()

    return df
