import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker: str, lookback_days: int = 30, interval: str = '1h'):
    df = yf.download(ticker, period=f"{lookback_days}d", interval=interval)
    df = df[df['Volume'] > 0]  # 去除盘前/夜盘无量数据
    df.dropna(inplace=True)
    return df


def generate_signals(df, window=20):
    df['high_rolling'] = df['High'].rolling(window).max()
    df['low_rolling'] = df['Low'].rolling(window).min()

    df['trade_signal'] = np.where(df['Close'] > df['high_rolling'].shift(1), 'buy',
                           np.where(df['Close'] < df['low_rolling'].shift(1), 'sell', np.nan))

    df['position'] = 0
    df.loc[df['trade_signal'] == 'buy', 'position'] = 1
    df.loc[df['trade_signal'] == 'sell', 'position'] = 0
    df['position'] = df['position'].ffill().fillna(0)

    df['returns'] = df['Close'].pct_change()
    df['strategy'] = df['position'].shift(1) * df['returns']
    df['equity_curve'] = (1 + df['strategy']).cumprod()

    return df

def simulate_trades(df, initial_cash=10000):
    position = 0
    cash = initial_cash
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        price = row['Close']
        signal = row.get('trade_signal')

        if signal == 'buy' and cash >= price:
            qty = int(cash / price)
            cost = qty * price
            cash -= cost
            position += qty
            trades.append({'time': idx, 'action': 'buy', 'price': price, 'qty': qty})

        elif signal == 'sell' and position > 0:
            proceeds = position * price
            cash += proceeds
            trades.append({'time': idx, 'action': 'sell', 'price': price, 'qty': position})
            position = 0

        equity = cash + position * price
        equity_curve.append({'time': idx, 'equity': equity})

    df['equity'] = pd.Series({e['time']: e['equity'] for e in equity_curve})
    trades_df = pd.DataFrame(trades)

    # 盈亏计算
    pnl_list = []
    buy_stack = []
    for trade in trades:
        if trade['action'] == 'buy':
            buy_stack.append(trade)
        elif trade['action'] == 'sell' and buy_stack:
            entry = buy_stack.pop(0)
            pnl = (trade['price'] - entry['price']) / entry['price']
            pnl_list.append({
                'buy_time': entry['time'],
                'sell_time': trade['time'],
                'buy_price': entry['price'],
                'sell_price': trade['price'],
                'return': round(pnl * 100, 2)
            })

    pnl_df = pd.DataFrame(pnl_list)
    return df, trades_df, pnl_df
