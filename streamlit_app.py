import streamlit as st
import plotly.graph_objs as go
from backtest import load_data, breakout_strategy

st.title("📈 Breakout 策略回测")
symbol = st.text_input("股票代码", value="AAPL")
period = st.selectbox("周期", ['1mo', '3mo', '6mo'], index=1)
interval = st.selectbox("K线周期", ['5m', '15m', '30m', '1h'], index=1)

@st.cache_data
def get_data(symbol, period, interval):
    df = load_data(symbol, period, interval)
    return breakout_strategy(df)

df = get_data(symbol, period, interval)

# K线图 + 信号标记
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="K线"
))

# 买入信号
buy_signals = df[df['Signal'] == 'Buy']
fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals['Close'],
    mode='markers',
    marker=dict(color='green', size=10, symbol='triangle-up'),
    name='Buy'
))

# 卖出信号 + 盈亏标注
sell_signals = df[df['Signal'] == 'Sell']
fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals['Close'],
    mode='markers+text',
    marker=dict(color='red', size=10, symbol='triangle-down'),
    text=[f"{p:.1%}" for p in sell_signals['ProfitPct']],
    textposition='top center',
    name='Sell'
))

fig.update_layout(title=f"{symbol} Breakout 策略", xaxis_rangeslider_visible=True)

st.plotly_chart(fig, use_container_width=True)

# 策略累计收益曲线
st.subheader("📊 策略累计收益")
cum_fig = go.Figure()
cum_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['CumulativeReturn'],
    mode='lines',
    name='Cumulative Return'
))
cum_fig.update_layout(yaxis_title="累积收益", xaxis_title="时间")
st.plotly_chart(cum_fig, use_container_width=True)
