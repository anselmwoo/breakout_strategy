import streamlit as st
from backtest import fetch_data, breakout_strategy
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config(layout="wide")
st.title("短线突破策略回测与交易信号展示")

# ---------------- 辅助函数 ----------------
def sharpe_ratio(returns, freq_per_year=252):
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret == 0:
        return 0
    return (mean_ret / std_ret) * np.sqrt(freq_per_year)

def max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def batch_backtest(df, param_grid, include_short):
    results = []
    for params in param_grid:
        rsi_w, ema_s, ema_l = params
        df_, _ = breakout_strategy(df.copy(), rsi_w, ema_s, ema_l, include_short)
        equity = df_['equity_curve']
        total_return = equity.iloc[-1] - 1
        returns = df_['strategy_returns'].dropna()
        sharpe = sharpe_ratio(returns)
        mdd = max_drawdown(equity)
        results.append({
            'RSI窗口': rsi_w,
            'EMA短期窗口': ema_s,
            'EMA长期窗口': ema_l,
            '总收益率': total_return,
            '夏普率': sharpe,
            '最大回撤': mdd
        })
    return pd.DataFrame(results)

# ---------------- Sidebar 参数设置 ------------------
with st.sidebar:
    st.header("基础参数设置")
    ticker = st.text_input("股票代码 (Ticker)", "RCAT")
    lookback_days = st.slider("回测天数", 7, 30, 14)
    interval = st.selectbox("时间间隔", options=['5m', '15m', '1h'], index=1)
    rsi_window = st.slider("RSI 窗口", 7, 21, 14)
    ema_short_window = st.slider("短期EMA窗口", 5, 20, 9)
    ema_long_window = st.slider("长期EMA窗口", 10, 50, 21)
    include_short = st.checkbox("计算做空收益（双向策略）", value=True)

    st.header("批量回测参数区间设置（选填）")
    rsi_min = st.number_input("RSI窗口最小值", min_value=1, max_value=50, value=14)
    rsi_max = st.number_input("RSI窗口最大值", min_value=1, max_value=50, value=16)
    rsi_step = st.number_input("RSI窗口步长", min_value=1, max_value=10, value=1)

    ema_short_min = st.number_input("EMA短期窗口最小值", min_value=1, max_value=50, value=9)
    ema_short_max = st.number_input("EMA短期窗口最大值", min_value=1, max_value=50, value=11)
    ema_short_step = st.number_input("EMA短期窗口步长", min_value=1, max_value=10, value=1)

    ema_long_min = st.number_input("EMA长期窗口最小值", min_value=1, max_value=100, value=21)
    ema_long_max = st.number_input("EMA长期窗口最大值", min_value=1, max_value=100, value=25)
    ema_long_step = st.number_input("EMA长期窗口步长", min_value=1, max_value=20, value=1)

    run_batch = st.button("开始批量回测")

# 根据回测天数确定period参数
if lookback_days <= 7:
    periods = ['7d']
elif lookback_days <= 14:
    periods = ['14d']
else:
    periods = ['30d']

st.write(f"尝试获取股票 {ticker}，周期设置为：{periods}，时间间隔：{interval}")

# ---------------- 数据获取与策略计算 ------------------
try:
    df = fetch_data(ticker, periods=periods, intervals=[interval])
    df, trades = breakout_strategy(df, rsi_window, ema_short_window, ema_long_window, include_short=include_short)
    df.index = pd.to_datetime(df.index)

    mpf_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    mpf_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 时间滑动条，控制图表显示时间区间
    start_date = df.index.min().to_pydatetime()
    end_date = df.index.max().to_pydatetime()
    selected_range = st.slider("选择显示时间段（用于图表）",
                               min_value=start_date,
                               max_value=end_date,
                               value=(start_date, end_date),
                               format="MM/DD HH:mm")

    filtered_df = df.loc[(df.index >= selected_range[0]) & (df.index <= selected_range[1])]
    filtered_mpf_df = mpf_df.loc[filtered_df.index]

    # 买卖信号点
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

    # 布局分栏
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
        st.subheader("策略累计收益曲线与持有收益对比")

        # 策略累计收益（选时间段内，排除无成交量时间）
        strategy_equity = filtered_df[filtered_df['volume'] > 0]['equity_curve']
        strategy_equity = strategy_equity.reset_index().rename(columns={'equity_curve': '累计收益'})

        # 持有收益 = 当前收盘价 / 首日收盘价
        hold_return = filtered_df['close'] / filtered_df['close'].iloc[0]
        hold_return = hold_return.reset_index().rename(columns={'close': '持有收益'})

        # 合并数据
        df_melt = pd.merge(strategy_equity, hold_return, on='datetime')
        df_melt = df_melt.melt(id_vars=['datetime'], value_vars=['累计收益', '持有收益'],
                               var_name='策略类型', value_name='收益')

        # 画线
        chart = (
            alt.Chart(df_melt)
            .mark_line()
            .encode(
                x='datetime:T',
                y=alt.Y('收益:Q', scale=alt.Scale(zero=False)),
                color=alt.Color('策略类型:N',
                                scale=alt.Scale(domain=['累计收益', '持有收益'],
                                                range=['#1f77b4', '#ff7f0e']))
            )
            .properties(
                width=600,
                height=400,
                title='策略累计收益 vs 持有收益对比'
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # 交易信号表
    st.subheader("所有交易信号（含盈亏）")
    trades_display = trades[['datetime', 'trade_type', 'trade_price', 'pnl']].copy()
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

    # ---------------- 批量回测部分 ----------------
    if run_batch:
        try:
            rsi_values = list(range(rsi_min, rsi_max + 1, rsi_step))
            ema_short_values = list(range(ema_short_min, ema_short_max + 1, ema_short_step))
            ema_long_values = list(range(ema_long_min, ema_long_max + 1, ema_long_step))

            param_grid = [
                (rsi_w, ema_s, ema_l)
                for rsi_w in rsi_values
                for ema_s in ema_short_values
                for ema_l in ema_long_values
                if ema_s < ema_l
            ]

            if not param_grid:
                st.warning("无有效参数组合，请调整参数区间")
            else:
                st.info(f"开始批量回测，参数组合数量: {len(param_grid)}")
                results_df = batch_backtest(df, param_grid, include_short)

                # 排序取Top5
                top5 = results_df.sort_values(by='总收益率', ascending=False).head(5)

                # 评分函数
                def score_series(s, reverse=False):
                    vmin, vmax = s.min(), s.max()
                    if vmax == vmin:
                        return pd.Series(50, index=s.index)
                    norm = (s - vmin) / (vmax - vmin)
                    return (1 - norm if reverse else norm) * 100

                top5['收益得分'] = score_series(top5['总收益率'])
                top5['夏普得分'] = score_series(top5['夏普率'])
                top5['回撤得分'] = score_series(top5['最大回撤'], reverse=True)
                top5['综合得分'] = (top5['收益得分'] + top5['夏普得分'] + top5['回撤得分']) / 3
                top5 = top5.sort_values(by='综合得分', ascending=False)

                st.subheader("批量回测 Top 5 参数组合")
                st.dataframe(top5.style.format({
                    '总收益率': '{:.2%}',
                    '夏普率': '{:.2f}',
                    '最大回撤': '{:.2%}',
                    '收益得分': '{:.1f}',
                    '夏普得分': '{:.1f}',
                    '回撤得分': '{:.1f}',
                    '综合得分': '{:.1f}',
                }))
        except Exception as e:
            st.error(f"批量回测出错: {e}")

except Exception as e:
    st.error(f"运行出错: {type(e)}\n{e}")
