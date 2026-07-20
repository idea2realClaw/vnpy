"""
run_vix_ratio_thr1_6.py —— 测试「VIX_ma3 / VIX_ma30 二值 high_cash」阈值改 1.6 的方案

合成：ratio = VIX_ma3 / VIX_ma30（无百分位窗口，抗 2020 污染）
规则：high_cash → ratio > thr 时空仓，否则满仓 SPY
对比：thr=1.6（本次） vs thr=2.0（原收益最优） vs SPY 买入持有
窗口：全样本 2011-2025 / 样本外 2016-2026

用法（仓库根目录）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ratio_thr1_6.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"


def nav_for(thr, direction, ratio, ret, valid):
    n = len(ret)
    pos = np.ones(n)
    for t in range(n):
        if not valid[t] or np.isnan(ratio[t]):
            continue
        if direction == "high_cash":
            pos[t] = 0.0 if ratio[t] > thr else 1.0
        else:
            pos[t] = 0.0 if ratio[t] < thr else 1.0
    nav = np.ones(n + 1)
    for t in range(n):
        nav[t + 1] = nav[t] * (1.0 + pos[t] * ret[t])
    return nav, pos


def slice_metrics(nav, dates, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    if len(ii) < 2:
        return None
    seg = nav[ii[0]:ii[-1] + 1]
    yrs = (np.datetime64(d1) - np.datetime64(d0)) / np.timedelta64(1, "D") / 365.25
    tot = seg[-1] / seg[0] - 1.0
    peak = np.maximum.accumulate(seg)
    mdd = (seg / peak - 1.0).min()
    cagr = seg[-1] ** (1.0 / yrs) - 1.0 if yrs > 0 else 0.0
    rd = tot / abs(mdd) if mdd < 0 else 0.0
    # 真实年化夏普：用策略每日收益（nav 差分）* sqrt(252)
    dr = seg[1:] / seg[:-1] - 1.0
    sd = dr.std(ddof=1)
    sharpe = (dr.mean() / sd) * np.sqrt(252) if sd > 0 else 0.0
    return tot, cagr, mdd, sharpe, rd


def main():
    vix = load_close("VIX")
    spy = load_close("SPY")
    idx = vix.index.intersection(spy.index)
    vix, spy = vix.loc[idx], spy.loc[idx]
    dates = idx.to_numpy()
    s = spy.to_numpy(float)
    ret = np.diff(s) / s[:-1]
    v = vix.to_numpy(float)

    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().to_numpy()
    ratio = ma3 / ma30

    n = len(ret)
    dt = dates[:n]
    valid = ~np.isnan(ratio[:n]) & (dt >= np.datetime64("2011-01-01"))

    # 三种方案
    nav_16, pos_16 = nav_for(1.6, "high_cash", ratio, ret, valid)
    nav_20, pos_20 = nav_for(2.0, "high_cash", ratio, ret, valid)
    spy_nav = np.concatenate([[1.0], np.cumprod(1.0 + ret)])

    cash16 = float(np.mean(pos_16 == 0.0))
    cash20 = float(np.mean(pos_20 == 0.0))

    WIN = {"全样本 2011-2025": ("2011-01-01", "2025-12-31"),
           "样本外 2016-2026": ("2016-01-01", "2026-07-15")}

    print("=== VIX 比值二值 high_cash：thr=1.6 vs thr=2.0 vs SPY ===")
    rows = []
    for wname, (d0, d1) in WIN.items():
        m16 = slice_metrics(nav_16, dates, d0, d1)
        m20 = slice_metrics(nav_20, dates, d0, d1)
        ms = slice_metrics(spy_nav, dates, d0, d1)
        print(f"\n[{wname}]")
        print(f"  thr=1.6 : 收益={m16[0]*100:7.1f}%  回撤={m16[2]*100:6.1f}%  夏普={m16[3]:.2f}")
        print(f"  thr=2.0 : 收益={m20[0]*100:7.1f}%  回撤={m20[2]*100:6.1f}%  夏普={m20[3]:.2f}")
        print(f"  SPY     : 收益={ms[0]*100:7.1f}%  回撤={ms[2]*100:6.1f}%  夏普={ms[3]:.2f}")
        rows.append((wname, m16[0]*100, m16[2]*100, m16[3],
                     m20[0]*100, m20[2]*100, m20[3],
                     ms[0]*100, ms[2]*100, ms[3]))
    print(f"\n空仓占比：thr=1.6 = {cash16*100:.1f}%  |  thr=2.0 = {cash20*100:.1f}%")

    # ---- HTML：两面板净值 ----
    def seg_of(nav, d0, d1):
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        seg = nav[ii[0]:ii[-1] + 1]
        return seg / seg[0] * 100, dates[ii[0]:ii[-1] + 1]

    fig = go.Figure()
    titles = list(WIN.keys())
    specs = [[{"type": "xy"}]] * 2
    # 用 subplots 手动：先建两个 trace 用 xaxis/yaxis
    # 为简单，建两个子图布局
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"{titles[0]}（起点=100）", f"{titles[1]}（起点=100）"),
                        vertical_spacing=0.12, row_heights=[0.5, 0.5])

    for r, (wname, (d0, d1)) in enumerate(WIN.items(), start=1):
        s16, d16 = seg_of(nav_16, d0, d1)
        s20, _ = seg_of(nav_20, d0, d1)
        ss, _ = seg_of(spy_nav, d0, d1)
        fig.add_trace(go.Scatter(x=d16, y=s16, name="thr=1.6",
                                 line=dict(color="#d62728", width=2.2)), row=r, col=1)
        fig.add_trace(go.Scatter(x=d16, y=s20, name="thr=2.0",
                                 line=dict(color="#ff7f0e", width=2.0, dash="dot")), row=r, col=1)
        fig.add_trace(go.Scatter(x=d16, y=ss, name="SPY 买入持有",
                                 line=dict(color="#1f77b4", width=1.6)), row=r, col=1)

    fig.update_layout(
        title="VIX_ma3/VIX_ma30 二值 high_cash：thr=1.6 vs thr=2.0 vs SPY",
        template="plotly_white", height=920,
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0),
        margin=dict(t=70, b=70),
    )
    fig.update_yaxes(title_text="净值 (起点=100)", row=1, col=1)
    fig.update_yaxes(title_text="净值 (起点=100)", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    path = f"{OUT_DIR}/vix_ratio_thr1_6.html"
    plot_offline(fig, filename=path, auto_open=False)

    # ---- ratio 子图（1.6 与 2.0 阈值线）----
    ri_m = (dates >= np.datetime64("2011-01-01")) & (dates <= np.datetime64("2026-07-15"))
    ri_seg = ratio[ri_m]
    ri_dt = dates[ri_m]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ri_dt, y=ri_seg, name="ratio=ma3/ma30",
                              line=dict(color="#2ca02c", width=1.2)))
    fig2.add_hline(y=1.6, line=dict(color="#d62728", dash="dash"),
                   annotation_text="thr=1.6 → 空仓", annotation_position="top right")
    fig2.add_hline(y=2.0, line=dict(color="#ff7f0e", dash="dash"),
                   annotation_text="thr=2.0 → 空仓", annotation_position="bottom right")
    fig2.update_layout(title="合成 ratio 与阈值线（thr=1.6 / 2.0）", xaxis_title="日期",
                       yaxis_title="VIX_ma3 / VIX_ma30", template="plotly_white", height=380)
    path_ind = f"{OUT_DIR}/vix_ratio_thr1_6_indicator.html"
    plot_offline(fig2, filename=path_ind, auto_open=False)

    # ---- CSV ----
    df = pd.DataFrame(rows, columns=["窗口", "thr1.6_收益%", "thr1.6_回撤%", "thr1.6_夏普",
                                     "thr2.0_收益%", "thr2.0_回撤%", "thr2.0_夏普",
                                     "SPY_收益%", "SPY_回撤%", "SPY_夏普"])
    df.to_csv(f"{OUT_DIR}/vix_ratio_thr1_6.csv", index=False)

    print(f"\n交付：{path}\n      {path_ind}\n      {OUT_DIR}/vix_ratio_thr1_6.csv")


if __name__ == "__main__":
    main()
