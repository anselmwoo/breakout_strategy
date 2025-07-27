import streamlit as st
from backtest import fetch_data, generate_signals, simulate_trades
import plotly.graph_objs as go

st.set_page_config(layout="wide")

st.sidebar.title("策略参数设置")
ticker = st.sidebar.text_input("股票代码", "RCAT")
lookback = st.sidebar.slider("回看天数", 5, 60, 30)
interval = st.sidebar.selectbox("时间粒度", ['1h', '30m', '15m'])

df = fetch_data(ticker, lookback, interval)
df = generate_signals(df)
df, trades_df, pnl_df = simulate_trades(df)

# K线图 + 信号标记
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='K线'))

# 买入信号
buy_signals = df[df['trade_signal'] == 'buy']
fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals['Close'],
    mode='markers',
    name='Buy',
    marker=dict(symbol='triangle-up', color='green', size=10)
))

# 卖出信号
sell_signals = df[df['trade_signal'] == 'sell']
fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals['Close'],
    mode='markers',
    name='Sell',
    marker=dict(symbol='triangle-down', color='red', size=10)
))

fig.update_layout(
    title=f"{ticker} K线图（含买卖信号）",
    xaxis_rangeslider_visible=True,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# 收益曲线
st.subheader("策略累计收益")
equity_fig = go.Figure()
equity_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['equity'],
    mode='lines',
    name='Equity Curve'
))
equity_fig.update_layout(height=400, xaxis_title='时间', yaxis_title='策略总资产')
st.plotly_chart(equity_fig, use_container_width=True)

# 每笔交易盈亏
st.subheader("每笔交易明细（含盈亏比例）")
st.dataframe(pnl_df)
