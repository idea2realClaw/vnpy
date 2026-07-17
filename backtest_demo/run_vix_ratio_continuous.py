"""
run_vix_ratio_continuous.py —— 连续仓位规则 C=(2-ratio)*2 测试（夹 0~100%）

合成 ratio = VIX_ma3 / VIX_ma30（同前）。
连续仓位：C[t] = clip((2 - ratio[t]) * 2, 0, 1)
  · ratio >= 2.0  -> C=0（空仓，恐慌尖峰）
  · ratio = 1.5   -> C=1.0（满仓）
  · ratio < 1.5   -> C=1.0（满仓，夹顶）
  · 1.5<ratio<2.0 -> 线性减仓（如 1.75 -> 50%）
日收益 = C[t] * ret[t]；净值递推。

对比：SPY 买入持有、二值 thr=2.0(high_cash)。
窗口：全样本 2011-2025、样本外 2016-2026。
指标：总收益(%)、最大回撤(%)、年化夏普(无风险=0)、平均仓位(%)。

用法（仓库根目录执行）：
    PYTHONPATH=/Users/zhuxiaodong/Documents/GitRepo/vnpy \
      /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ratio_continuous.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"
WINS = {"全样本 2011-2025": ("2011-01-01", "2025-12-31"),
        "样本外 2016-2026": ("2016-01-01", "2026-07-15")}


def nav_binary(thr, ratio, ret, valid):
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


def nav_cont(ratio, ret, valid):
    n = len(ret)
    c = np.ones(n)
    for t in range(n):
        if not valid[t] or np.isnan(ratio[t]):
            continue
        c[t] = min(max((2.0 - ratio[t]) * 2.0, 0.0), 1.0)
    nav = np.ones(n + 1)
    for t in range(n):
        nav[t + 1] = nav[t] * (1.0 + c[t] * ret[t])
    return nav, c


def metrics(nav, pos_or_c, ret, dates, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    seg = nav[ii[0]:ii[-1] + 1]
    tot = seg[-1] / seg[0] - 1.0
    peak = np.maximum.accumulate(seg)
    mdd = (seg / peak - 1.0).min()
    tidx = np.arange(ii[0], ii[-1])
    day_ret = pos_or_c[tidx] * ret[tidx]
    sharpe = day_ret.mean() / day_ret.std(ddof=1) * np.sqrt(252) if day_ret.std(ddof=1) > 0 else 0.0
    avg_pos = float(np.mean(pos_or_c[tidx])) * 100
    return tot * 100, mdd * 100, sharpe, avg_pos


def spy_metrics(s, ret, dates, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    tot = (s[ii[-1]] / s[ii[0]] - 1.0) * 100
    peak = np.maximum.accumulate(s[ii[0]:ii[-1] + 1])
    mdd = ((s[ii[0]:ii[-1] + 1] / peak) - 1.0).min() * 100
    tidx = np.arange(ii[0], ii[-1])
    dr = ret[tidx]
    shp = dr.mean() / dr.std(ddof=1) * np.sqrt(252)
    return tot, mdd, shp, 100.0


def seg_of(nav, dates, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    seg = nav[ii[0]:ii[-1] + 1]
    return seg / seg[0], dates[ii[0]:ii[-1] + 1]


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

    nav_c, c = nav_cont(ratio, ret, valid)
    nav_b, _ = nav_binary(2.0, ratio, ret, valid)

    rows = []
    for wname, (d0, d1) in WINS.items():
        ct, cm, cs, cp = metrics(nav_c, c, ret, dates, d0, d1)
        bt, bm, bs, bp = metrics(nav_b, np.where(ratio[:len(ret)] > 2.0, 0.0, 1.0), ret, dates, d0, d1)
        st, sm, ss, sp = spy_metrics(s, ret, dates, d0, d1)
        rows.append({
            "窗口": wname,
            "连续C收益%": round(ct, 1), "连续C回撤%": round(cm, 1),
            "连续C夏普": round(cs, 2), "连续C仓位%": round(cp, 1),
            "二值2.0收益%": round(bt, 1), "二值2.0回撤%": round(bm, 1), "二值2.0夏普": round(bs, 2),
            "SPY收益%": round(st, 1), "SPY回撤%": round(sm, 1), "SPY夏普": round(ss, 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\n连续规则仓位公式: C = clip((2 - ratio)*2, 0, 1)  | 平均仓位: "
          f"全样本 {metrics(nav_c,c,ret,dates,'2011-01-01','2025-12-31')[3]:.1f}%  "
          f"样本外 {metrics(nav_c,c,ret,dates,'2016-01-01','2026-07-15')[3]:.1f}%")
    df.to_csv(f"{OUT_DIR}/vix_ratio_continuous.csv", index=False)

    # ---- 双面板图：连续 vs 二值2.0 vs SPY ----
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.13,
                        subplot_titles=(
        f"全样本 2011-2025 ｜ 连续 +{df.iloc[0]['连续C收益%']}% / 二值2.0 +{df.iloc[0]['二值2.0收益%']}% / SPY +{df.iloc[0]['SPY收益%']}%",
        f"样本外 2016-2026 ｜ 连续 +{df.iloc[1]['连续C收益%']}% / 二值2.0 +{df.iloc[1]['二值2.0收益%']}% / SPY +{df.iloc[1]['SPY收益%']}%",
    ))
    for r, (d0, d1) in enumerate(WINS.values(), start=1):
        sc, dc = seg_of(nav_c, dates, d0, d1)
        sb, _ = seg_of(nav_b, dates, d0, d1)
        sspy, _ = seg_of(np.concatenate([[1.0], np.cumprod(1.0 + ret)]), dates, d0, d1)
        fig.add_trace(go.Scatter(x=dc, y=sc * 100, name="连续 C=(2-ratio)*2",
                                 legendgroup="c", line=dict(color="#d62728", width=2.5)), row=r, col=1)
        fig.add_trace(go.Scatter(x=dc, y=sb * 100, name="二值 thr=2.0",
                                 legendgroup="b", showlegend=(r == 1),
                                 line=dict(color="#ff7f0e", width=1.8, dash="dot")), row=r, col=1)
        fig.add_trace(go.Scatter(x=dc, y=sspy * 100, name="SPY 买入持有",
                                 legendgroup="s", showlegend=(r == 1),
                                 line=dict(color="#1f77b4", width=2)), row=r, col=1)
        fig.update_yaxes(title_text="净值 (起点=100)", row=r, col=1)
        fig.update_xaxes(title_text="日期", row=r, col=1)

    fig.update_layout(
        title="VIX_ma3/VIX_ma30 比值 · 连续仓位 vs 二值 vs SPY",
        template="plotly_white", height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    path = f"{OUT_DIR}/vix_ratio_continuous.html"
    plot_offline(fig, filename=path, auto_open=False)
    print(f"\n交付：{path}\n      {OUT_DIR}/vix_ratio_continuous.csv")


if __name__ == "__main__":
    main()
