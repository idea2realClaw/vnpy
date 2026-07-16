# coding=utf-8
# -*- coding: utf-8 -*-
"""用真实 vnpy BacktestingEngine 测试 VixRankStrategy（纯 VIX 满仓/空仓 SPY）。

- 对 rank_mode ∈ {p2 短期, p1 长期, yyi 阴阳} × threshold ∈ {0.6,0.7,0.8,0.9} 做参数扫描；
- 与 SPY 买入持有基准对比；
- 输出 plotly 对比图 + 全扫描汇总 CSV。

所有决策仅用「截至当日」的 VIX（on_init 全量载入，无未来函数）；成交在下根 bar 开盘撮合。
OOS 窗口与之前一致：2022-01-03 起。
"""
import os
import sqlite3
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from vnpy.trader.constant import Interval
from vnpy.trader.object import HistoryRequest
from vnpy_ctastrategy.backtesting import BacktestingEngine

from vix_rank_strategy import VixRankStrategy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_PATH = os.path.join(OUT_DIR, "vix_strategy_chart.html")
SUMMARY_PATH = os.path.join(OUT_DIR, "vix_strategy_summary.csv")
DB_PATH = os.path.expanduser("~/.vntrader/database.db")

VT_SYMBOL = "SPY.SMART"
TEST_START = dt.datetime(2022, 1, 1)
TEST_END = dt.datetime(2026, 7, 16)
OOS_DATE = "2022-01-03"   # 样本外起点（此日之前不交易）
CAPITAL = 1_000_000

RANK_MODES = {"短期p2": "p2", "长期p1": "p1", "阴阳yyi": "yyi"}
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]


def load_spy_buyhold():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT datetime, close_price FROM dbbardata WHERE symbol='SPY' AND exchange='SMART' "
        "AND datetime >= '2022-01-01' ORDER BY datetime",
        con,
    )
    con.close()
    df["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date")["close_price"].astype(float)
    df = df[df.index >= pd.Timestamp(OOS_DATE)]
    nav = df / df.iloc[0] * 100.0
    return df, nav


def run_one(rank_mode, threshold):
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=VT_SYMBOL,
        interval=Interval.DAILY,
        start=TEST_START,
        rate=0.0,
        slippage=0.0,
        size=1.0,
        pricetick=0.01,
        capital=CAPITAL,
        end=TEST_END,
    )
    setting = {
        "feature_symbol": "VIX",
        "feature_exchange": "SMART",
        "rank_mode": rank_mode,
        "threshold": threshold,
        "use_full_capital": True,
        "trade_start": OOS_DATE,
    }
    engine.add_strategy(VixRankStrategy, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    df = engine.daily_df
    if df is None or df.empty:
        return None, None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[df.index >= pd.Timestamp(OOS_DATE)]
    nav = df["balance"] / df["balance"].iloc[0] * 100.0
    # 指标
    bal = df["balance"].values
    total_ret = bal[-1] / bal[0] - 1.0
    years = (df.index[-1] - df.index[0]).days / 365.0
    cagr = (bal[-1] / bal[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    peak = np.maximum.accumulate(bal)
    mdd = float(((bal - peak) / peak).min())
    return nav, {
        "total_return_%": round(total_ret * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "max_drawdown_%": round(mdd * 100, 2),
        "trades": engine.trade_count,
        "final_balance": round(float(bal[-1]), 0),
    }


def main():
    print("加载 SPY 买入持有基准 ...")
    spy_df, spy_nav = load_spy_buyhold()
    spy_bh = spy_nav.iloc[-1] / 100.0 - 1.0
    print(f"SPY 买入持有: {spy_bh*100:.2f}%  (窗口 {spy_nav.index[0].date()}~{spy_nav.index[-1].date()})")

    rows = []
    best = {}
    for mname, mode in RANK_MODES.items():
        best_nav, best_cfg = None, None
        for thr in THRESHOLDS:
            nav, stats = run_one(mode, thr)
            if nav is None:
                print(f"  [{mname} thr={thr}] 无结果，跳过")
                continue
            rows.append({
                "mode": mname, "rank_mode": mode, "threshold": thr,
                **stats,
            })
            print(f"  [{mname} thr={thr}] 收益 {stats['total_return_%']:.2f}%  "
                  f"回撤 {stats['max_drawdown_%']:.2f}%  笔数 {stats['trades']}")
            if best_nav is None or nav.iloc[-1] > best_nav.iloc[-1]:
                best_nav, best_cfg = nav, (thr, stats)
        best[mname] = (best_nav, best_cfg)

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n汇总已保存到 {SUMMARY_PATH}")

    # ---- 绘图 ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spy_nav.index, y=spy_nav.values, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2.5)))
    colors = {"短期p2": "#2ca02c", "长期p1": "#d62728", "阴阳yyi": "#9467bd"}
    for mname, (nav, (thr, stats)) in best.items():
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values,
            name=f"{mname} 满仓/空仓(thr={thr:.2f}, +{stats['total_return_%']:.1f}%)",
            line=dict(color=colors[mname], width=2.2)))

    fig.update_layout(
        title="VIX Rank 满仓/空仓 SPY 策略（真实引擎回测, 2022-01-03 起, 起点=100）",
        xaxis_title="日期", yaxis_title="净值 (起点=100)",
        template="plotly_white", height=640,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    plot_offline(fig, filename=CHART_PATH, auto_open=False)
    print(f"图表已保存到 {CHART_PATH}")

    print("\n=== 关键结论（真实引擎）===")
    print(f"SPY 买入持有: {spy_bh*100:.2f}%")
    for mname, (nav, (thr, stats)) in best.items():
        print(f"  {mname}: 最佳 thr={thr:.2f} -> {stats['total_return_%']:.2f}%, "
              f"回撤 {stats['max_drawdown_%']:.2f}%, {stats['trades']} 笔")


if __name__ == "__main__":
    main()
