import streamlit as st
from backtest import fetch_data, breakout_strategy
import mplfinance as mpf
import pandas as pd

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

# periods 根据 lookback_days 设定
if lookback_days <= 7:
    periods = ['7d']
elif lookback_days <= 14:
    periods = ['14d']
else:
    periods = ['30d']

st.write(f"尝试获取股票 {ticker}，周期设置为：{periods}，时间间隔：{interval}")

try:
    # 获取数据，强制只用用户选的interval，改fetch_data支持传入interval
    df = fetch_data(ticker, periods=periods, intervals=[interval])

    # 运行策略
    df, trades = breakout_strategy(df, rsi_window, ema_short_window, ema_long_window)

    # 删除夜盘和盘前空白：mplfinance 支持用 DataFrame 索引过滤
    # 直接用原始时间索引，不显示非交易时间
    df_for_plot = df.copy()
    df_for_plot.index = pd.to_datetime(df_for_plot.index)

    # mplfinance 要求列名首字母大写，重新构造df
    mpf_df = df_for_plot[['open', 'high', 'low', 'close', 'volume']].copy()
    mpf_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    st.subheader("K线图 (去除夜盘空白)")
    fig, ax = mpf.plot(mpf_df,
                       type='candle',
                       style='yahoo',
                       mav=(ema_short_window, ema_long_window),
                       volume=True,
                       returnfig=True,
                       datetime_format='%m-%d %H:%M')
    st.pyplot(fig)

    st.subheader("所有交易信号")
    trades_display = trades[['datetime', 'trade_type', 'trade_price']].copy()
    trades_display['datetime'] = trades_display['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    trades_display = trades_display.rename(columns={
        'datetime': '交易时间',
        'trade_type': '交易类型',
        'trade_price': '交易价格'
    })
    st.dataframe(trades_display.reset_index(drop=True))

except Exception as e:
    st.error(f"运行出错: {e}")
