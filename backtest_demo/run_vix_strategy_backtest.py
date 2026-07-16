# coding=utf-8
# -*- coding: utf-8 -*-
"""用真实 vnpy BacktestingEngine 测试 VixRankStrategy（纯 VIX 满仓/空仓 SPY）。

覆盖两类框架：
  【A. 阴阳指数框架】rank_mode ∈ {p2 短期, p1 长期, yyi 阴阳} × threshold ∈ {0.6,0.7,0.8,0.9}
  【B. 恐慌反转框架（改进 1+2）】panic_reversal × 多组 (spike_metric, spike_thr, fall_days)
    - 尖峰检测用 z-score（VIX 相对自身 60 日历史）或绝对水平；
    - 连续 fall_days 日回落确认恐慌反转，sign 翻转（高位尖峰=买入信号）。

与 SPY 买入持有基准对比；输出 plotly 对比图 + 全扫描汇总 CSV。

同时跑两个窗口：
  - 2022-2026（与之前结果可比，偏牛市，择时难跑赢 B&H）
  - 2017-2026（含 2018 Volmageddon、2020 新冠崩盘，公平验证「降回撤」价值）

所有决策仅用「截至当日」的 VIX（on_init 全量载入，无未来函数）；成交在下根 bar 开盘撮合。
"""
import os
import sqlite3
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from vnpy.trader.constant import Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

from vix_rank_strategy import VixRankStrategy

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.expanduser("~/.vntrader/database.db")

VT_SYMBOL = "SPY.SMART"
CAPITAL = 1_000_000

# 阴阳指数框架扫描
RANK_MODES = {"短期p2": "p2", "长期p1": "p1", "阴阳yyi": "yyi"}
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

# 恐慌反转框架扫描（改进 1+2）
PANIC_CONFIGS = [
    ("panic_z2.0_f2", {"rank_mode": "panic_reversal", "spike_metric": "z",     "spike_thr": 2.0, "fall_days": 2}),
    ("panic_z2.5_f2", {"rank_mode": "panic_reversal", "spike_metric": "z",     "spike_thr": 2.5, "fall_days": 2}),
    ("panic_z2.5_f3", {"rank_mode": "panic_reversal", "spike_metric": "z",     "spike_thr": 2.5, "fall_days": 3}),
    ("panic_z3.0_f3", {"rank_mode": "panic_reversal", "spike_metric": "z",     "spike_thr": 3.0, "fall_days": 3}),
    ("panic_lv28_f2", {"rank_mode": "panic_reversal", "spike_metric": "level", "spike_thr": 28.0, "fall_days": 2}),
    ("panic_lv30_f2", {"rank_mode": "panic_reversal", "spike_metric": "level", "spike_thr": 30.0, "fall_days": 2}),
    ("panic_lv30_f3", {"rank_mode": "panic_reversal", "spike_metric": "level", "spike_thr": 30.0, "fall_days": 3}),
    ("panic_lv35_f3", {"rank_mode": "panic_reversal", "spike_metric": "level", "spike_thr": 35.0, "fall_days": 3}),
]

# 两个测试窗口
WINDOWS = {
    "2022_2026": {"test_start": dt.datetime(2022, 1, 1),  "oos": "2022-01-03"},
    "2017_2026": {"test_start": dt.datetime(2016, 1, 1),  "oos": "2017-01-03"},  # 含 2018/2020 崩盘
}
TEST_END = dt.datetime(2026, 7, 16)


def load_spy_buyhold(oos_date):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT datetime, close_price FROM dbbardata WHERE symbol='SPY' AND exchange='SMART' "
        "AND datetime >= '2016-01-01' ORDER BY datetime",
        con,
    )
    con.close()
    df["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date")["close_price"].astype(float)
    df = df[df.index >= pd.Timestamp(oos_date)]
    nav = df / df.iloc[0] * 100.0
    return df, nav


def run_one(rank_mode, threshold, test_start, oos_date, extra=None):
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=VT_SYMBOL,
        interval=Interval.DAILY,
        start=test_start,
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
        "trade_start": oos_date,
    }
    if extra:
        setting.update(extra)
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
    df = df[df.index >= pd.Timestamp(oos_date)]
    nav = df["balance"] / df["balance"].iloc[0] * 100.0
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


def run_window(win_name, cfg):
    test_start = cfg["test_start"]
    oos_date = cfg["oos"]
    label = win_name.replace("_", "-")
    chart_path = os.path.join(OUT_DIR, f"vix_strategy_chart_{win_name}.html")
    summary_path = os.path.join(OUT_DIR, f"vix_strategy_summary_{win_name}.csv")

    print(f"\n########## 窗口 {label}（OOS {oos_date} ~ 2026-07-15）##########")
    print("加载 SPY 买入持有基准 ...")
    spy_df, spy_nav = load_spy_buyhold(oos_date)
    spy_bh = spy_nav.iloc[-1] / 100.0 - 1.0
    print(f"SPY 买入持有: {spy_bh*100:.2f}%  (窗口 {spy_nav.index[0].date()}~{spy_nav.index[-1].date()})")

    rows = []
    best = {}

    for mname, mode in RANK_MODES.items():
        best_nav, best_cfg = None, None
        for thr in THRESHOLDS:
            nav, stats = run_one(mode, thr, test_start, oos_date)
            if nav is None:
                continue
            rows.append({"group": "阴阳指数", "config": mname, "rank_mode": mode,
                         "threshold": thr, **stats})
            if best_nav is None or nav.iloc[-1] > best_nav.iloc[-1]:
                best_nav, best_cfg = nav, (thr, stats)
        best[mname] = (best_nav, best_cfg)

    panic_navs = {}
    for cname, pcfg in PANIC_CONFIGS:
        nav, stats = run_one("panic_reversal", 0.0, test_start, oos_date, extra=pcfg)
        if nav is None:
            continue
        rows.append({"group": "恐慌反转", "config": cname, "rank_mode": "panic_reversal",
                     "spike_metric": pcfg["spike_metric"], "spike_thr": pcfg["spike_thr"],
                     "fall_days": pcfg["fall_days"], **stats})
        panic_navs[cname] = (nav, stats)

    best_ret_cfg = max(panic_navs.items(), key=lambda kv: kv[1][0].iloc[-1])
    # 回撤是负值，max() 取「最不负面」= 回撤最小 = 最优
    best_mdd_cfg = max(panic_navs.items(), key=lambda kv: kv[1][1]["max_drawdown_%"])

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    print(f"汇总已保存到 {summary_path}")

    # ---- 绘图 ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spy_nav.index, y=spy_nav.values, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2.5)))
    colors_yy = {"短期p2": "#2ca02c", "长期p1": "#d62728", "阴阳yyi": "#9467bd"}
    for mname, (nav, (thr, stats)) in best.items():
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values,
            name=f"阴阳 {mname}(thr={thr:.2f}, +{stats['total_return_%']:.1f}%)",
            line=dict(color=colors_yy[mname], width=1.8, dash="dot")))
    cname_ret, (nav_ret, stats_ret) = best_ret_cfg
    fig.add_trace(go.Scatter(
        x=nav_ret.index, y=nav_ret.values,
        name=f"恐慌反转 最佳收益 {cname_ret}(+{stats_ret['total_return_%']:.1f}%, DD{stats_ret['max_drawdown_%']:.1f}%)",
        line=dict(color="#ff7f0e", width=2.6)))
    if best_mdd_cfg[0] != cname_ret:
        cname_mdd, (nav_mdd, stats_mdd) = best_mdd_cfg
        fig.add_trace(go.Scatter(
            x=nav_mdd.index, y=nav_mdd.values,
            name=f"恐慌反转 最佳回撤 {cname_mdd}(+{stats_mdd['total_return_%']:.1f}%, DD{stats_mdd['max_drawdown_%']:.1f}%)",
            line=dict(color="#17becf", width=2.2, dash="dash")))

    fig.update_layout(
        title=f"VIX 满仓/空仓 SPY 策略（真实引擎回测, {label}, 起点=100）",
        xaxis_title="日期", yaxis_title="净值 (起点=100)",
        template="plotly_white", height=660,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    plot_offline(fig, filename=chart_path, auto_open=False)
    print(f"图表已保存到 {chart_path}")

    spy_peak = spy_nav.cummax()
    spy_dd = float(((spy_nav - spy_peak) / spy_peak).min()) * 100
    print(f"=== 窗口 {label} 关键结论 ===")
    print(f"SPY 买入持有: {spy_bh*100:.2f}%  峰值回撤 {spy_dd:.2f}%")
    for mname, (nav, (thr, stats)) in best.items():
        print(f"  阴阳 {mname}: 最佳 thr={thr:.2f} -> {stats['total_return_%']:.2f}%, "
              f"回撤 {stats['max_drawdown_%']:.2f}%, {stats['trades']} 笔")
    cname_ret, (nav_ret, stats_ret) = best_ret_cfg
    print(f"  恐慌反转 最佳收益: {cname_ret} -> +{stats_ret['total_return_%']:.2f}%, "
          f"回撤 {stats_ret['max_drawdown_%']:.2f}%, {stats_ret['trades']} 笔")
    cname_mdd, (nav_mdd, stats_mdd) = best_mdd_cfg
    print(f"  恐慌反转 最佳回撤: {cname_mdd} -> +{stats_mdd['total_return_%']:.2f}%, "
          f"回撤 {stats_mdd['max_drawdown_%']:.2f}%, {stats_mdd['trades']} 笔")
    return spy_bh


def main():
    for win_name, cfg in WINDOWS.items():
        run_window(win_name, cfg)


if __name__ == "__main__":
    main()
