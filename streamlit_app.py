import streamlit as st
from backtest import fetch_data, breakout_strategy
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config(layout="wide")
st.title("短线突破策略回测与交易信号展示")

# ---------------- Sidebar 参数设置 ------------------
with st.sidebar:
    ticker = st.text_input("股票代码 (Ticker)", "RCAT")
    lookback_days = st.slider("回测天数", 7, 30, 14)
    interval = st.selectbox("时间间隔", options=['5m', '15m', '1h'], index=1)
    rsi_window = st.slider("RSI 窗口", 7, 21, 14)
    ema_short_window = st.slider("短期EMA窗口", 5, 20, 9)
    ema_long_window = st.slider("长期EMA窗口", 10, 50, 21)
    include_short = st.checkbox("计算做空收益（双向策略）", value=True)  # 复选框，默认勾选

if lookback_days <= 7:
    periods = ['7d']
elif lookback_days <= 14:
    periods = ['14d']
else:
    periods = ['30d']

st.write(f"尝试获取股票 {ticker}，周期设置为：{periods}，时间间隔：{interval}")

# ---------------- 数据获取与策略 ------------------
try:
    df = fetch_data(ticker, periods=periods, intervals=[interval])
    df, trades = breakout_strategy(df, rsi_window, ema_short_window, ema_long_window)
    df.index = pd.to_datetime(df.index)
    mpf_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    mpf_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # ---------------- 时间滑动条 ------------------
    start_date = df.index.min().to_pydatetime()
    end_date = df.index.max().to_pydatetime()
    selected_range = st.slider("选择显示时间段（用于图表）",
                               min_value=start_date,
                               max_value=end_date,
                               value=(start_date, end_date),
                               format="MM/DD HH:mm")

    # 过滤数据用于图表绘制
    filtered_df = df.loc[(df.index >= selected_range[0]) & (df.index <= selected_range[1])]
    filtered_mpf_df = mpf_df.loc[filtered_df.index]

    # ---------------- 信号点处理 ------------------
    buy_signals = trades[trades['trade_type'] == 'Buy']
    sell_signals = trades[trades['trade_type'] == 'Sell']

    buys = pd.Series(data=np.nan, index=filtered_mpf_df.index)
    sells = pd.Series(data=np.nan, index=filtered_mpf_df.index)

    for _, row in buy_signals.iterrows():
        dt = pd.to_datetime(row['datetime'])
        if dt in buys.index:
            buys.at[dt] = df.loc[dt, 'low'] * 0.995

    for _, row in sell_signals.iterrows():
        dt = pd.to_datetime(row['datetime'])
        if dt in sells.index:
            sells.at[dt] = df.loc[dt, 'high'] * 1.005

    ap_buy = mpf.make_addplot(buys, type='scatter', markersize=100, marker='^', color='g')
    ap_sell = mpf.make_addplot(sells, type='scatter', markersize=100, marker='v', color='r')

    # ---------------- 布局：K线图与收益曲线左右分栏 ------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K线图 (带买卖信号标记)")
        fig, _ = mpf.plot(filtered_mpf_df,
                          type='candle',
                          style='yahoo',
                          mav=(ema_short_window, ema_long_window),
                          volume=True,
                          addplot=[ap_buy, ap_sell],
                          returnfig=True,
                          datetime_format='%m-%d %H:%M',
                          figsize=(12, 6))
        st.pyplot(fig)
    
    with col2:
        st.subheader("策略累计收益曲线（去除非交易时间段）")
        equity_curve_clean = df[df['volume'] > 0]['equity_curve']
        equity_curve_clean = df[df['volume'] > 0]['equity_curve'].reset_index()
        equity_curve_clean.columns = ['datetime', 'equity_curve']
        
        chart = (
            alt.Chart(equity_curve_clean)
            .mark_line()
            .encode(
                x='datetime:T',
                y=alt.Y('equity_curve:Q', scale=alt.Scale(zero=False))  # 这里告诉altair不强制0起点
            )
            .properties(
                width=600,
                height=400,
                title='策略累计收益曲线（Y轴自动适应，非零起点）'
            )
        )
        
        st.altair_chart(chart, use_container_width=True)


    # ---------------- 交易表格 ------------------
    st.subheader("所有交易信号（含盈亏）")
    trades_display = trades[['datetime', 'trade_type', 'trade_price', 'pnl']].copy()

    # 时间字段格式化（确保是 datetime）
    trades_display['datetime'] = pd.to_datetime(trades_display['datetime'], errors='coerce')
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
    st.error(f"运行出错: {type(e)}\n{e}")
