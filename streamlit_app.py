import streamlit as st
from backtest import fetch_data, breakout_strategy
import matplotlib.pyplot as plt

st.title("短线突破策略回测")

ticker = st.text_input("股票代码 (Ticker)", "RCAT")
lookback_days = st.number_input("回测天数", min_value=7, max_value=30, value=14)

# 根据回测天数决定periods列表
if lookback_days <= 7:
    periods = ['7d']
elif lookback_days <= 14:
    periods = ['14d']
else:
    periods = ['30d']

st.write(f"尝试获取股票 {ticker}，周期设置为：{periods}")

try:
    df = fetch_data(ticker, periods=periods)
    df = breakout_strategy(df)

    st.subheader("价格与均线走势")
    st.line_chart(df[['close', 'ema_short', 'ema_long']])

    st.subheader("策略累计收益曲线")
    st.line_chart(df['equity_curve'])

    st.subheader("策略最后几行数据")
    st.dataframe(df[['close', 'ema_short', 'ema_long', 'position', 'strategy_returns', 'equity_curve']].tail())

except Exception as e:
    st.error(f"运行出错: {e}")
