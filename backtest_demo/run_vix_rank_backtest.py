# coding=utf-8
# -*- coding: utf-8 -*-
"""
参考 /Users/zhuxiaodong/Documents/GitRepo/daofund 的 VIX 阴阳指数策略，
用「长期 VIX Rank」+「短期 VIX Rank」测试收益情况。

daofund 核心逻辑 (src/dao.py):
  percentage_rank(val, series) = (series 中严格小于 val 的个数)/len(series)
  p1 = percentrank(VIX_today, 近10年VIX)          # 长期 rank
  p2 = percentrank(VIX_today, 近50日VIX)           # 短期 rank
  p3 = 1 - percentrank(VIX_today, 近2000日VIX)     # 长期 rank 取反
  yyi = index = 0.7*p2 + 0.15*p1 + 0.15*p3         # 阴阳指数

本脚本:
  1. 从 vnpy SQLite 读取 VIX / SPY / TLT 日线 (已含 2016-2026)。
  2. 逐日复刻上述 percentrank（仅用当日及之前的历史，无未来函数）。
  3. 用 rank 信号做「SPY↔TLT 轮动」与「SPY↔现金 择时」两类测试，
     对比 SPY / TLT 买入持有基准。
  4. 输出汇总 CSV + plotly HTML 对比图。

OOS 窗口与之前一致：2022-01-03 ~ 今。
"""
import os
import sqlite3
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

DB_PATH = os.path.expanduser("~/.vntrader/database.db")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_PATH = os.path.join(OUT_DIR, "vix_rank_chart.html")
SUMMARY_PATH = os.path.join(OUT_DIR, "vix_rank_summary.csv")

# 回测窗口（样本外，与之前 RF 实验一致）
TEST_START = "2022-01-03"
# 长期窗口 ~10 年交易日；短期 50 日；p3 用 2000 日
LONG_WINDOW = 2520
SHORT_WINDOW = 50
P3_WINDOW = 2000


def load_close(symbol, exchange):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT datetime, close_price FROM dbbardata WHERE symbol='{symbol}' AND exchange='{exchange}' ORDER BY datetime",
        con,
    )
    con.close()
    df["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date")["close_price"].astype(float)
    return df


def percent_rank_growing(values, window):
    """逐日 percentrank：对 t，rank = (values[max(0,t-window):t] 中 < values[t] 的比例)。
    与 daofund percentage_rank 一致（严格小于）。"""
    n = len(values)
    out = np.full(n, np.nan)
    for t in range(1, n):
        lo = max(0, t - window)
        w = values[lo:t]
        if len(w) == 0:
            continue
        out[t] = float(np.mean(w < values[t]))
    return out


def build_ranks(vix):
    """输入: 按日对齐的 VIX Series（索引为 date）。返回三个 rank 的 Series（同索引）。"""
    v = vix.to_numpy()
    p1 = percent_rank_growing(v, LONG_WINDOW)      # 长期 rank
    p2 = percent_rank_growing(v, SHORT_WINDOW)     # 短期 rank
    p3 = 1.0 - percent_rank_growing(v, P3_WINDOW)  # 长期取反
    idx = vix.index
    p1 = pd.Series(p1, index=idx)
    p2 = pd.Series(p2, index=idx)
    yyi = 0.7 * p2 + 0.15 * p1 + 0.15 * p3
    return p1, p2, p3, yyi


def simulate_rotation(signal, spy_ret, tlt_ret, threshold, cash=False):
    """根据 signal 与 threshold 做每日轮动。
    signal[t] < threshold -> 持 SPY；否则 -> 持 TLT（cash=True 时持现金）。
    用当日收盘信号决定次日仓位（避免未来函数）。
    返回净值 Series（与 signal 同索引）。"""
    nav = pd.Series(np.nan, index=signal.index)
    nav.iloc[0] = 1.0
    pos = "SPY"
    for t in range(1, len(signal)):
        # 用前一日的信号决定今日仓位
        s = signal.iloc[t - 1]
        if np.isnan(s):
            pos = "SPY"  # 信号未就绪时默认满仓风险资产
        else:
            pos = "SPY" if s < threshold else ("CASH" if cash else "TLT")
        r = spy_ret.iloc[t] if pos == "SPY" else (0.0 if pos == "CASH" else tlt_ret.iloc[t])
        nav.iloc[t] = nav.iloc[t - 1] * (1.0 + r)
    return nav


def max_drawdown(nav):
    peak = nav.cummax()
    return float(((nav - peak) / peak).min())


def cagr(nav, years):
    total = nav.iloc[-1] / nav.iloc[0]
    if years <= 0 or total <= 0:
        return 0.0
    return float((total ** (1.0 / years)) - 1.0)


def main():
    print("加载数据 ...")
    vix = load_close("VIX", "SMART")
    spy = load_close("SPY", "SMART")
    tlt = load_close("TLT", "SMART")

    # 对齐到共同交易日
    common = vix.index.intersection(spy.index).intersection(tlt.index)
    vix = vix.loc[common]
    spy = spy.loc[common]
    tlt = tlt.loc[common]

    p1, p2, p3, yyi = build_ranks(vix)

    # 收益率序列
    spy_ret = spy.pct_change().fillna(0.0)
    tlt_ret = tlt.pct_change().fillna(0.0)

    # 仅取测试窗口（rank 仍用全历史计算，无未来函数）
    mask = (spy.index >= TEST_START)
    idx = spy.index[mask]
    spy_r = spy_ret.loc[idx]
    tlt_r = tlt_ret.loc[idx]
    p1_t = p1.loc[idx]
    p2_t = p2.loc[idx]
    yyi_t = yyi.loc[idx]

    n_days = len(idx)
    years = (idx[-1] - idx[0]).days / 365.0
    print(f"测试窗口 {idx[0].date()} ~ {idx[-1].date()}  ({n_days} 交易日, {years:.2f} 年)")

    # 基准
    spy_nav = (1.0 + spy_r).cumprod()
    tlt_nav = (1.0 + tlt_r).cumprod()
    spy_bh = spy_nav.iloc[-1] - 1.0
    tlt_bh = tlt_nav.iloc[-1] - 1.0

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    modes = {
        "长期rank(p1)": p1_t,
        "短期rank(p2)": p2_t,
        "阴阳yyi(0.7p2+0.15p1+0.15p3)": yyi_t,
    }

    rows = []
    best = {}        # mode -> (thr, tot, nav)  SPY↔TLT 最佳
    best_cash = {}   # mode -> (thr, tot, nav)  SPY↔现金 最佳
    for mname, sig in modes.items():
        best_nav = None
        best_cfg = None
        best_cash_nav = None
        best_cash_cfg = None
        for thr in thresholds:
            nav = simulate_rotation(sig, spy_r, tlt_r, thr, cash=False)
            tot = nav.iloc[-1] - 1.0
            dd = max_drawdown(nav)
            cg = cagr(nav, years)
            # 统计换仓次数 & 在 SPY 天数
            pos = (sig < thr).astype(int)  # 1=SPY, 0=TLT
            switches = int((pos.diff().abs().fillna(0) > 0).sum())
            days_spy = int(pos.sum())
            rows.append({
                "mode": mname, "threshold": thr, "variant": "SPY↔TLT",
                "total_return_%": round(tot * 100, 2),
                "CAGR_%": round(cg * 100, 2),
                "max_drawdown_%": round(dd * 100, 2),
                "switches": switches,
                "days_in_SPY": days_spy,
            })
            if best_nav is None or nav.iloc[-1] > best_nav.iloc[-1]:
                best_nav = nav
                best_cfg = (thr, tot, nav)
            # 现金择时
            navc = simulate_rotation(sig, spy_r, tlt_r, thr, cash=True)
            totc = navc.iloc[-1] - 1.0
            posc = (sig < thr).astype(int)
            rowsc = int((posc.diff().abs().fillna(0) > 0).sum())
            daysc = int(posc.sum())
            rows.append({
                "mode": mname, "threshold": thr, "variant": "SPY↔现金",
                "total_return_%": round(totc * 100, 2),
                "CAGR_%": round(cagr(navc, years) * 100, 2),
                "max_drawdown_%": round(max_drawdown(navc) * 100, 2),
                "switches": rowsc,
                "days_in_SPY": daysc,
            })
            if best_cash_nav is None or navc.iloc[-1] > best_cash_nav.iloc[-1]:
                best_cash_nav = navc
                best_cash_cfg = (thr, totc, navc)
        best[mname] = best_cfg
        best_cash[mname] = best_cash_cfg

    # 另测「SPY↔现金」择时（downside protection 视角），用各 mode 最佳 threshold
    for mname, sig in modes.items():
        thr = best[mname][0]
        nav = simulate_rotation(sig, spy_r, tlt_r, thr, cash=True)
        tot = nav.iloc[-1] - 1.0
        rows.append({
            "mode": mname, "threshold": thr, "variant": "SPY↔现金",
            "total_return_%": round(tot * 100, 2),
            "CAGR_%": round(cagr(nav, years) * 100, 2),
            "max_drawdown_%": round(max_drawdown(nav) * 100, 2),
            "switches": int((((sig < thr).astype(int)).diff().abs().fillna(0) > 0).sum()),
            "days_in_SPY": int((sig < thr).sum()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n汇总已保存到 {SUMMARY_PATH}")
    print(summary.to_string(index=False))

    # ---- 绘图 ----
    fig = go.Figure()
    # 基准
    fig.add_trace(go.Scatter(x=idx, y=spy_nav.values * 100, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2.5)))
    fig.add_trace(go.Scatter(x=idx, y=tlt_nav.values * 100, name="TLT 买入持有",
                             line=dict(color="#ff7f0e", width=2, dash="dot")))

    colors = {
        "长期rank(p1)": "#d62728",       # 红=涨(区域惯例)
        "短期rank(p2)": "#2ca02c",       # 绿=跌
        "阴阳yyi(0.7p2+0.15p1+0.15p3)": "#9467bd",
    }
    # 各 mode 最佳「轮动」线（展示 SPY↔TLT 表现）
    for mname, (thr, tot, nav) in best.items():
        fig.add_trace(go.Scatter(
            x=idx, y=nav.values * 100,
            name=f"{mname} 轮动(thr={thr:.2f}, +{tot*100:.1f}%)",
            line=dict(color=colors[mname], width=1.8, dash="dot")))

    # 各 mode 最佳「现金择时」线（展示 downside protection 价值）
    for mname, (thr, tot, nav) in best_cash.items():
        fig.add_trace(go.Scatter(
            x=idx, y=nav.values * 100,
            name=f"{mname} 择时现金(thr={thr:.2f}, +{tot*100:.1f}%)",
            line=dict(color=colors[mname], width=2.2)))

    fig.update_layout(
        title="VIX 长期/短期 Rank 策略收益对比 (2022-01-03 起, 起点=100)",
        xaxis_title="日期", yaxis_title="净值 (起点=100)",
        template="plotly_white", height=640,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    plot_offline(fig, filename=CHART_PATH, auto_open=False)
    print(f"\n图表已保存到 {CHART_PATH}")

    print("\n=== 关键结论 ===")
    print(f"SPY 买入持有: {spy_bh*100:.2f}%  | TLT 买入持有: {tlt_bh*100:.2f}%")
    print("-- SPY↔TLT 轮动最佳 --")
    for mname, (thr, tot, nav) in best.items():
        print(f"  {mname}: thr={thr:.2f} -> {tot*100:.2f}%, 最大回撤 {max_drawdown(nav)*100:.2f}%")
    print("-- SPY↔现金 择时最佳 --")
    for mname, (thr, tot, nav) in best_cash.items():
        print(f"  {mname}: thr={thr:.2f} -> {tot*100:.2f}%, 最大回撤 {max_drawdown(nav)*100:.2f}%")


if __name__ == "__main__":
    main()
