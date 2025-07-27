import streamlit as st
from backtest import fetch_data, breakout_strategy
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("短线突破策略回测与交易信号展示")

# 侧边栏参数
with st.sidebar:
    ticker = st.text_input("股票代码 (Ticker)", "RCAT")
    lookback_days = st.slider("回测天数", 7, 30, 14)
    interval = st.selectbox("时间间隔", options=['5m', '15m', '1h'], index=1)
    rsi_window = st.slider("RSI 窗口", 7, 21, 14)
    ema_short_window = st.slider("短期EMA窗口", 5, 20, 9)
    ema_long_window = st.slider("长期EMA窗口", 10, 50, 21)

if lookback_days <= 7:
    periods = ['7d']
elif lookback_days <= 14:
    periods = ['14d']
else:
    periods = ['30d']

st.write(f"尝试获取股票 {ticker}，周期设置为：{periods}，时间间隔：{interval}")

try:
    df = fetch_data(ticker, periods=periods, intervals=[interval])
    df, trades = breakout_strategy(df, rsi_window, ema_short_window, ema_long_window)

    df.index = pd.to_datetime(df.index)
    mpf_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    mpf_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 准备收益曲线数据
    ap = mpf.make_addplot(df['equity_curve'], panel=1, color='g', ylabel='Equity')

    st.subheader("K线图 (去除夜盘空白) + 策略累计收益")
    fig, axlist = mpf.plot(mpf_df,
                           type='candle',
                           style='yahoo',
                           mav=(ema_short_window, ema_long_window),
                           volume=True,
                           addplot=ap,
                           panel_ratios=(3,1),
                           returnfig=True,
                           datetime_format='%m-%d %H:%M')
    st.pyplot(fig)

    st.subheader("所有交易信号（含盈亏）")
    trades_display = trades[['datetime', 'trade_type', 'trade_price', 'pnl']].copy()
    trades_display['datetime'] = trades_display['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    trades_display['pnl'] = trades_display['pnl'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    trades_display = trades_display.rename(columns={
        'datetime': '交易时间',
        'trade_type': '交易类型',
        'trade_price': '交易价格',
        'pnl': '盈亏比例'
    })
    st.dataframe(trades_display.reset_index(drop=True))

except Exception as e:
    st.error(f"运行出错: {e}")
