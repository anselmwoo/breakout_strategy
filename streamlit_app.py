import streamlit as st
from backtest import fetch_data, compute_indicators, breakout_strategy
from config import stock_list, lookback_days

st.title("📈 Breakout 策略回测")

for ticker in stock_list:
    st.subheader(f"Ticker: {ticker}")
    df = fetch_data(ticker, lookback_days)
    df = compute_indicators(df)
    trades = breakout_strategy(df)

    for t in trades:
        st.write(t)