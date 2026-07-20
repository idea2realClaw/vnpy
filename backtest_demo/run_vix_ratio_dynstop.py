"""
run_vix_ratio_dynstop.py —— VIX_ma3/VIX_ma30 二值 + 动态止损

合成：ratio = VIX_ma3 / VIX_ma30
基础规则(high_cash)：ratio > thr -> 空仓，否则满仓 SPY
【新增动态止损】：策略净值自高点回撤 > 5% 时，强制空仓；
    - 至少空仓 5 天（冷却）
    - 冷却结束后，且 ratio < thr 才恢复满仓；若 ratio 仍 >= thr 则继续空仓
    （即「空仓5天，直到 ratio 小于阈值」）

对比：动态止损(thr=1.6 / 2.0) vs 原版无止损(thr=2.0) vs SPY 买入持有
窗口：全样本 2011-2025 / 样本外 2016-2026

用法（仓库根目录）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ratio_dynstop.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"


def nav_base(thr, ratio, ret, valid):
    """原版二值 high_cash：ratio>thr 空仓。"""
    n = len(ret)
    pos = np.ones(n)
    for t in range(n):
        if not valid[t] or np.isnan(ratio[t]):
            continue
        pos[t] = 0.0 if ratio[t] > thr else 1.0
    nav = np.ones(n + 1)
    for t in range(n):
        nav[t + 1] = nav[t] * (1.0 + pos[t] * ret[t])
    return nav, pos


def nav_dynstop(thr, ratio, ret, valid, dd_trig=0.05, cool=5):
    """二值 + 动态止损：高点回撤>dd_trig 强制空仓，最少 cool 天，直到 ratio<thr 才恢复。

    注意：
    1) 峰值(high water mark)从策略生效首日(第一个 valid 日, 即 2011-01-01)起重新计，
       避免 2006-2010 预热期(含 2008 金融危机)的峰值污染回撤计算导致永久空仓。
    2) 退出强制空仓的当日用 just_exited 守卫，禁止同日重新触发，否则会因峰值被冻结
       在崩盘前高位、nav 始终低于它而陷入"退出即触发"的死循环（永远空仓）。
    """
    n = len(ret)
    pos = np.ones(n)
    nav = [1.0]
    peak = 1.0
    forced = False
    cd = 0
    started = False
    just_exited = False
    for t in range(n):
        if not valid[t] or np.isnan(ratio[t]):
            pos[t] = 1.0  # 预热期满仓 SPY（与基线一致）
        else:
            if not started:
                peak = 1.0  # 策略生效首日重置峰值，回撤只从 2011 起算
                started = True
            if forced:
                pos[t] = 0.0
                cd -= 1
                if cd <= 0 and ratio[t] < thr:
                    forced = False
                    just_exited = True  # 本日刚退出，禁止同日重新触发
            else:
                pos[t] = 0.0 if ratio[t] > thr else 1.0
        nav.append(nav[-1] * (1.0 + pos[t] * ret[t]))
        if started:
            peak = max(peak, nav[-1])
            dd = nav[-1] / peak - 1.0
            if not forced and not just_exited and dd < -dd_trig:
                forced = True
                cd = cool
                pos[t] = 0.0  # 触发当日也转空仓
        just_exited = False
    return np.array(nav), pos


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
    sharpe = 0.0
    dr = seg[1:] / seg[:-1] - 1.0
    if len(dr) > 1:
        sd = dr.std(ddof=1)
        sharpe = (dr.mean() / sd) * np.sqrt(252) if sd > 0 else 0.0
    return tot, mdd, sharpe


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

    # 方案
    nav_spy = np.concatenate([[1.0], np.cumprod(1.0 + ret)])
    nav_b20, _ = nav_base(2.0, ratio, ret, valid)
    nav_s20, _ = nav_dynstop(2.0, ratio, ret, valid)
    nav_s16, _ = nav_dynstop(1.6, ratio, ret, valid)

    WIN = {"全样本 2011-2025": ("2011-01-01", "2025-12-31"),
           "样本外 2016-2026": ("2016-01-01", "2026-07-15")}

    print("=== VIX 比值二值 + 动态止损(回撤>5%空仓≥5天, 直到 ratio<thr) ===")
    rows = []
    for wname, (d0, d1) in WIN.items():
        r = {}
        for name, nv in [("SPY", nav_spy), ("原版 thr=2.0", nav_b20),
                         ("止损 thr=2.0", nav_s20), ("止损 thr=1.6", nav_s16)]:
            r[name] = slice_metrics(nv, dates, d0, d1)
        print(f"\n[{wname}]")
        for name in ["SPY", "原版 thr=2.0", "止损 thr=2.0", "止损 thr=1.6"]:
            tot, mdd, sh = r[name]
            print(f"  {name:12s}: 收益={tot*100:7.1f}%  回撤={mdd*100:6.1f}%  夏普={sh:.2f}")
        rows.append((wname,
                     r["止损 thr=2.0"][0]*100, r["止损 thr=2.0"][1]*100, r["止损 thr=2.0"][2],
                     r["止损 thr=1.6"][0]*100, r["止损 thr=1.6"][1]*100, r["止损 thr=1.6"][2],
                     r["原版 thr=2.0"][0]*100, r["原版 thr=2.0"][1]*100, r["原版 thr=2.0"][2],
                     r["SPY"][0]*100, r["SPY"][1]*100, r["SPY"][2]))

    # ---- HTML 双面板 ----
    def seg_of(nav, d0, d1):
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        seg = nav[ii[0]:ii[-1] + 1]
        return seg / seg[0] * 100, dates[ii[0]:ii[-1] + 1]

    titles = list(WIN.keys())
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"{titles[0]}（起点=100）", f"{titles[1]}（起点=100）"),
                        vertical_spacing=0.13, row_heights=[0.5, 0.5])
    series = [("SPY 买入持有", nav_spy, "#1f77b4", 1.6),
              ("原版 thr=2.0 (无止损)", nav_b20, "#7f7f7f", 1.4),
              ("动态止损 thr=2.0", nav_s20, "#ff7f0e", 2.2),
              ("动态止损 thr=1.6", nav_s16, "#d62728", 2.0)]
    for r, (wname, (d0, d1)) in enumerate(WIN.items(), start=1):
        for label, nv, color, w in series:
            seg, dd = seg_of(nv, d0, d1)
            fig.add_trace(go.Scatter(x=dd, y=seg, name=label,
                                     line=dict(color=color, width=w)), row=r, col=1)
    fig.update_layout(
        title="VIX 比值二值 + 动态止损（回撤>5%空仓≥5天，直到 ratio<thr）",
        template="plotly_white", height=940,
        legend=dict(orientation="h", yanchor="top", y=-0.07, xanchor="left", x=0),
        margin=dict(t=70, b=80),
    )
    fig.update_yaxes(title_text="净值 (起点=100)", row=1, col=1)
    fig.update_yaxes(title_text="净值 (起点=100)", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    path = f"{OUT_DIR}/vix_ratio_dynstop.html"
    plot_offline(fig, filename=path, auto_open=False)

    # ratio 子图（阈值线 + 提示）
    ri_m = (dates >= np.datetime64("2011-01-01")) & (dates <= np.datetime64("2026-07-15"))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dates[ri_m], y=ratio[ri_m], name="ratio=ma3/ma30",
                              line=dict(color="#2ca02c", width=1.2)))
    fig2.add_hline(y=2.0, line=dict(color="#ff7f0e", dash="dash"),
                   annotation_text="thr=2.0", annotation_position="top right")
    fig2.add_hline(y=1.6, line=dict(color="#d62728", dash="dash"),
                   annotation_text="thr=1.6", annotation_position="bottom right")
    fig2.update_layout(title="合成 ratio 与阈值线（动态止损方案）", xaxis_title="日期",
                       yaxis_title="VIX_ma3 / VIX_ma30", template="plotly_white", height=380)
    path_ind = f"{OUT_DIR}/vix_ratio_dynstop_indicator.html"
    plot_offline(fig2, filename=path_ind, auto_open=False)

    # CSV
    df = pd.DataFrame(rows, columns=[
        "窗口", "止损2.0_收益%", "止损2.0_回撤%", "止损2.0_夏普",
        "止损1.6_收益%", "止损1.6_回撤%", "止损1.6_夏普",
        "原版2.0_收益%", "原版2.0_回撤%", "原版2.0_夏普",
        "SPY_收益%", "SPY_回撤%", "SPY_夏普"])
    df.to_csv(f"{OUT_DIR}/vix_ratio_dynstop.csv", index=False)

    print(f"\n交付：{path}\n      {path_ind}\n      {OUT_DIR}/vix_ratio_dynstop.csv")


if __name__ == "__main__":
    main()
