import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from itertools import product

st.set_page_config(layout="wide")

# 计算最大回撤
def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

# 计算夏普率
def sharpe_ratio(returns):
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)

# 策略回测函数
def backtest_strategy(df, threshold, ma_window, include_short=True):
    df = df.copy()
    df['ma'] = df['close'].rolling(window=ma_window).mean()
    df.dropna(inplace=True)

    df['signal'] = 0
    df.loc[df['close'] > df['ma'] * (1 + threshold), 'signal'] = 1
    if include_short:
        df.loc[df['close'] < df['ma'] * (1 - threshold), 'signal'] = -1

    df['position'] = df['signal'].shift().fillna(0)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['strategy'] = df['position'] * df['returns']

    df['equity'] = (1 + df['strategy']).cumprod()
    df['buy_hold'] = (1 + df['returns']).cumprod()

    return df

# 下载数据
@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.columns = [col.lower() for col in df.columns]
    return df

# 绘制图表
def plot_kline(df, title, buy_idx=None, sell_idx=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='K线'))

    if buy_idx is not None:
        fig.add_trace(go.Scatter(x=df['datetime'].iloc[buy_idx],
                                 y=df['close'].iloc[buy_idx],
                                 mode='markers',
                                 marker=dict(color='green', size=8),
                                 name='买入'))

    if sell_idx is not None:
        fig.add_trace(go.Scatter(x=df['datetime'].iloc[sell_idx],
                                 y=df['close'].iloc[sell_idx],
                                 mode='markers',
                                 marker=dict(color='red', size=8),
                                 name='卖出'))

    fig.update_layout(title=title, xaxis_rangeslider_visible=True, height=500)
    return fig

def plot_equity(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['equity'], name="策略收益", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['buy_hold'], name="持有收益", line=dict(color='orange')))
    
    y_data = pd.concat([df['equity'], df['buy_hold']])
    y_min, y_max = y_data.min(), y_data.max()
    y_margin = (y_max - y_min) * 0.05
    fig.update_layout(title='累计收益对比', yaxis_range=[y_min - y_margin, y_max + y_margin], height=500)
    return fig

# Streamlit 左侧参数
st.sidebar.header("参数设置")
symbol = st.sidebar.text_input("股票代码", value="RCAT")
period = st.sidebar.selectbox("周期", ['60d', '90d', '120d'], index=0)
interval = st.sidebar.selectbox("K线粒度", ['1h', '30m', '15m'], index=0)

# 是否包括空头
include_short = st.sidebar.checkbox("包括做空策略", value=True)

# 回测参数输入
mode = st.sidebar.radio("模式", ["单次回测", "批量回测"])

df = load_data(symbol, period, interval)

if mode == "单次回测":
    threshold = st.sidebar.slider("突破阈值 (%)", 0.1, 5.0, 1.0) / 100
    ma_window = st.sidebar.slider("均线窗口", 5, 60, 20)
    result_df = backtest_strategy(df, threshold, ma_window, include_short)

    col1, col2 = st.columns(2)
    with col1:
        buy_signals = result_df[result_df['signal'] == 1].index
        sell_signals = result_df[result_df['signal'] == -1].index if include_short else []
        st.plotly_chart(plot_kline(result_df, "价格与信号", buy_signals, sell_signals), use_container_width=True)
    with col2:
        st.plotly_chart(plot_equity(result_df), use_container_width=True)

    final_return = result_df['equity'].iloc[-1] - 1
    sharpe = sharpe_ratio(result_df['strategy'])
    mdd = max_drawdown(result_df['equity'])

    st.write(f"**最终收益率**: {final_return:.2%}")
    st.write(f"**夏普率**: {sharpe:.2f}")
    st.write(f"**最大回撤**: {mdd:.2%}")

else:
    st.sidebar.subheader("参数区间设置")
    th_min, th_max, th_step = st.sidebar.slider("阈值范围 (%)", 0.1, 5.0, (0.5, 2.0))  # 输入为百分比
    ma_min, ma_max, ma_step = st.sidebar.slider("均线窗口范围", 5, 60, (10, 30))
    th_values = np.arange(th_min / 100, th_max / 100 + 0.0001, th_step / 100)
    ma_values = range(ma_min, ma_max + 1, ma_step)

    results = []
    for th, ma in product(th_values, ma_values):
        try:
            r_df = backtest_strategy(df, th, ma, include_short)
            final_return = r_df['equity'].iloc[-1] - 1
            sr = sharpe_ratio(r_df['strategy'])
            dd = max_drawdown(r_df['equity'])
            results.append({
                'threshold': round(th, 4),
                'ma_window': ma,
                'return': final_return,
                'sharpe': sr,
                'max_drawdown': dd
            })
        except Exception as e:
            continue

    result_df = pd.DataFrame(results)
    result_df['score'] = result_df['return'] * result_df['sharpe'] / abs(result_df['max_drawdown'] + 1e-5)
    top5 = result_df.sort_values("score", ascending=False).head(5)
    st.subheader("Top 5 参数组合")
    st.dataframe(top5)

    best_params = top5.iloc[0]
    st.markdown(f"**最优参数组合回测**： threshold={best_params['threshold']}, ma_window={int(best_params['ma_window'])}")
    best_df = backtest_strategy(df, best_params['threshold'], int(best_params['ma_window']), include_short)

    col1, col2 = st.columns(2)
    with col1:
        buy_signals = best_df[best_df['signal'] == 1].index
        sell_signals = best_df[best_df['signal'] == -1].index if include_short else []
        st.plotly_chart(plot_kline(best_df, "价格与信号", buy_signals, sell_signals), use_container_width=True)
    with col2:
        st.plotly_chart(plot_equity(best_df), use_container_width=True)

