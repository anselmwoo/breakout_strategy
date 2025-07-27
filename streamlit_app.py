import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# 设置页面
st.set_page_config(page_title="策略回测平台", layout="wide")

# 技术指标函数
def apply_indicators(df, rsi_period, ema_period):
    df["RSI"] = RSIIndicator(close=df["Close"], window=rsi_period).rsi()
    df["EMA"] = EMAIndicator(close=df["Close"], window=ema_period).ema_indicator()
    return df

# 策略信号生成
def generate_signals(df, rsi_overbought, rsi_oversold):
    df["Signal"] = 0
    df["Signal"] = np.where(df["RSI"] < rsi_oversold, 1, df["Signal"])  # 买入
    df["Signal"] = np.where(df["RSI"] > rsi_overbought, -1, df["Signal"])  # 卖出
    df["Position"] = df["Signal"].shift().fillna(0)
    return df

# 策略回测计算
def backtest(df):
    df["Return"] = df["Close"].pct_change()
    df["Strategy"] = df["Return"] * df["Position"]
    df["Cumulative Return"] = (1 + df["Return"]).cumprod()
    df["Cumulative Strategy"] = (1 + df["Strategy"]).cumprod()

    # 风险指标
    sharpe = np.sqrt(252) * df["Strategy"].mean() / df["Strategy"].std() if df["Strategy"].std() != 0 else 0
    drawdown = (df["Cumulative Strategy"] / df["Cumulative Strategy"].cummax()) - 1
    max_drawdown = drawdown.min()

    return df, sharpe, max_drawdown

# 批量参数回测
def parameter_grid(rsi_range, ema_range, rsi_step, ema_step, rsi_oversold, rsi_overbought, data):
    results = []
    for rsi_p in range(rsi_range[0], rsi_range[1]+1, rsi_step):
        for ema_p in range(ema_range[0], ema_range[1]+1, ema_step):
            df_temp = data.copy()
            df_temp = apply_indicators(df_temp, rsi_p, ema_p)
            df_temp = generate_signals(df_temp, rsi_overbought, rsi_oversold)
            df_temp, sharpe, max_dd = backtest(df_temp)
            final_return = df_temp["Cumulative Strategy"].iloc[-1] - 1
            results.append({
                "RSI": rsi_p,
                "EMA": ema_p,
                "Return": final_return,
                "Sharpe": sharpe,
                "Max Drawdown": max_dd
            })
    return pd.DataFrame(results).sort_values("Return", ascending=False)

# 左侧栏：输入
st.sidebar.header("参数设置")

symbol = st.sidebar.text_input("股票代码", value="AMD")
start = st.sidebar.date_input("开始日期", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("结束日期", pd.to_datetime("today"))
interval = st.sidebar.selectbox("K线周期", ["1d", "1h", "15m", "5m"], index=0)

# 单次回测参数
st.sidebar.subheader("策略参数")
rsi_period = st.sidebar.number_input("RSI周期", 5, 50, 14)
ema_period = st.sidebar.number_input("EMA周期", 5, 100, 20)
rsi_oversold = st.sidebar.slider("超卖阈值", 0, 50, 30)
rsi_overbought = st.sidebar.slider("超买阈值", 50, 100, 70)

# 回测模式开关
batch_mode = st.sidebar.checkbox("启用批量参数回测")

if batch_mode:
    st.sidebar.subheader("参数区间设置")
    rsi_start = st.sidebar.number_input("RSI开始", 5, 50, 10)
    rsi_end = st.sidebar.number_input("RSI结束", 10, 80, 20)
    rsi_step = st.sidebar.number_input("RSI步长", 1, 20, 2)
    ema_start = st.sidebar.number_input("EMA开始", 5, 50, 10)
    ema_end = st.sidebar.number_input("EMA结束", 10, 100, 30)
    ema_step = st.sidebar.number_input("EMA步长", 1, 20, 5)

# 获取数据
@st.cache_data
def load_data(symbol, start_date, end_date, interval):
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df = df.dropna()
    # 确保列名规范
    df = df.rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low',
        'Adj Close': 'Close', 'Volume': 'Volume'
    })
    return df


df = load_data(symbol, start, end, interval)

st.title(f"{symbol} 策略回测可视化")

if batch_mode:
    st.subheader("📊 批量参数回测结果")
    result_df = parameter_grid(
        rsi_range=(rsi_start, rsi_end),
        ema_range=(ema_start, ema_end),
        rsi_step=rsi_step,
        ema_step=ema_step,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        data=df
    )
    st.dataframe(result_df.head(10).style.background_gradient(cmap="YlGn"))
    st.markdown("**Top 5 策略组合图：**")

    for idx, row in result_df.head(5).iterrows():
        st.markdown(f"**RSI: {row['RSI']} | EMA: {row['EMA']} | Return: {row['Return']:.2%} | Sharpe: {row['Sharpe']:.2f} | MaxDD: {row['Max Drawdown']:.2%}**")
else:
    df = apply_indicators(df, rsi_period, ema_period)
    df = generate_signals(df, rsi_overbought, rsi_oversold)
    df, sharpe, max_dd = backtest(df)

    st.subheader("📈 策略 vs 持有 收益曲线")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Cumulative Strategy"], label="策略收益", color="green")
    ax.plot(df.index, df["Cumulative Return"], label="持有收益", color="gray", linestyle="--")
    ax.set_ylabel("累计收益")
    ax.set_title(f"策略累计收益（Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2%}）")
    ax.legend()
    st.pyplot(fig)

    st.subheader("💹 K线图 + 信号")
    import plotly.graph_objects as go
    fig2 = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="K线")])
    buy_signals = df[df["Signal"] == 1]
    sell_signals = df[df["Signal"] == -1]
    fig2.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Close"],
                              mode='markers', marker=dict(color='green', size=8), name="买入"))
    fig2.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["Close"],
                              mode='markers', marker=dict(color='red', size=8), name="卖出"))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📄 交易明细")
    trade_log = df[df["Signal"] != 0][["Close", "Signal"]]
    st.dataframe(trade_log.tail(10))

