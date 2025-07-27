import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from backtest import generate_signals

# ---------------- Streamlit 配置 ----------------
st.set_page_config(layout='wide')
st.title("📈 Breakout 策略回测示例")

# ---------------- 参数设置 ----------------
symbol = st.sidebar.text_input("股票代码", "RCAT")
interval = st.sidebar.selectbox("时间间隔", ["1h", "30m", "15m"], index=0)
period = st.sidebar.selectbox("历史周期", ["30d", "60d", "90d"], index=0)
window = st.sidebar.slider("滚动窗口", 5, 50, 20)

# ---------------- 获取数据 ----------------
@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df = df[df.index.strftime('%H:%M:%S').between("09:30:00", "16:00:00")]  # 排除夜盘
    return df

df = load_data(symbol, period, interval)

if df.empty:
    st.warning("❌ 无法获取数据，请检查股票代码或网络。")
    st.stop()

# ---------------- 生成信号与回测 ----------------
df = generate_signals(df, window)

# ---------------- 绘制 K线图与信号 ----------------
fig = go.Figure()

# 添加 K 线
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='K线'))

# 添加买入点
buy_signals = df[df['trade_signal'] == 'buy']
fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals['Close'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='买入'))

# 添加卖出点
sell_signals = df[df['trade_signal'] == 'sell']
fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals['Close'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='卖出'))

fig.update_layout(title=f"{symbol} K线图（带买卖信号）",
                  xaxis_rangeslider_visible=True,
                  xaxis_title="时间", yaxis_title="价格",
                  height=600)
st.plotly_chart(fig, use_container_width=True)

# ---------------- 收益曲线图 ----------------
equity_fig = go.Figure()
equity_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['equity_curve'],
    mode='lines',
    name='策略收益',
    line=dict(color='blue')))
equity_fig.update_layout(title="策略累计收益曲线",
                         xaxis_title="时间",
                         yaxis_title="净值",
                         height=300)
st.plotly_chart(equity_fig, use_container_width=True)

# ---------------- 显示交易信号与盈亏 ----------------
trades = df[df['trade_signal'].isin(['buy', 'sell'])].copy()
trades['pct_change'] = trades['Close'].pct_change().shift(-1)
trades['pnl'] = trades['pct_change'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
st.subheader("📋 交易信号列表")
st.dataframe(trades[['trade_signal', 'Close', 'pnl']].dropna(), use_container_width=True)
