"""
run_vix_ratio_panels.py —— 把 VIX_ma3/VIX_ma30 二值 SPY 策略的两段收益曲线放进同一个 HTML 展示

展示内容：
  面板 1 —— 全样本 2011-2025：策略 vs SPY 买入持有（起点=100）
  面板 2 —— 样本外 2016-2026：策略 vs SPY 买入持有（起点=100）

训练所得最优参数（来自 run_vix_ratio.py）：方向=high_cash, 阈值 thr=2.00
  · 全样本 2011-2025：策略 +720.8% vs SPY +601.6%
  · 样本外 2016-2026：策略 +454.4% vs SPY +345.4%

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ratio_panels.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"

THR = 2.00            # 训练所得阈值
DIR = "high_cash"     # 训练所得方向


def nav_for(thr, direction, ratio, ret, valid):
    """二值 SPY：pos[t]=0(空仓)/1(满仓)。ratio[t] 为 t 日信号，作用于 ret[t]。"""
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


def seg_of(nav, dates, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    if len(ii) < 2:
        return None, None
    seg = nav[ii[0]:ii[-1] + 1]
    return seg / seg[0], dates[ii[0]:ii[-1] + 1]


def main():
    # ---- 数据 ----
    vix = load_close("VIX")
    spy = load_close("SPY")
    idx = vix.index.intersection(spy.index)
    vix = vix.loc[idx]
    spy = spy.loc[idx]
    dates = idx.to_numpy()
    s = spy.to_numpy(float)
    ret = np.diff(s) / s[:-1]
    v = vix.to_numpy(float)

    # ---- 平滑 + 比值 ----
    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().to_numpy()
    ratio = ma3 / ma30

    n = len(ret)
    dt = dates[:n]
    valid = ~np.isnan(ratio[:n]) & (dt >= np.datetime64("2011-01-01"))

    # ---- 净值 ----
    nav, _ = nav_for(THR, DIR, ratio, ret, valid)
    spy_nav = np.concatenate([[1.0], np.cumprod(1.0 + ret)])

    # ---- 两段切片（策略 / SPY，均起点=100）----
    win_full = ("2011-01-01", "2025-12-31")
    win_oos = ("2016-01-01", "2026-07-15")

    strat_full, d_full = seg_of(nav, dates, *win_full)
    spy_full, _ = seg_of(spy_nav, dates, *win_full)
    strat_oos, d_oos = seg_of(nav, dates, *win_oos)
    spy_oos, _ = seg_of(spy_nav, dates, *win_oos)

    full_ret = (strat_full[-1] / strat_full[0] - 1.0) * 100
    full_spy = (spy_full[-1] / spy_full[0] - 1.0) * 100
    oos_ret = (strat_oos[-1] / strat_oos[0] - 1.0) * 100
    oos_spy = (spy_oos[-1] / spy_oos[0] - 1.0) * 100

    # ---- 双面板 ----
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False, vertical_spacing=0.13,
        subplot_titles=(
            f"全样本 2011-2025　策略 +{full_ret:.1f}% ｜ SPY +{full_spy:.1f}%",
            f"样本外 2016-2026　策略 +{oos_ret:.1f}% ｜ SPY +{oos_spy:.1f}%",
        ),
    )

    # 面板 1
    fig.add_trace(go.Scatter(x=d_full, y=strat_full * 100,
                             name="VIX比值策略", legendgroup="strat",
                             line=dict(color="#d62728", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d_full, y=spy_full * 100,
                             name="SPY 买入持有", legendgroup="spy",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)

    # 面板 2
    fig.add_trace(go.Scatter(x=d_oos, y=strat_oos * 100,
                             name="VIX比值策略", legendgroup="strat", showlegend=False,
                             line=dict(color="#d62728", width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d_oos, y=spy_oos * 100,
                             name="SPY 买入持有", legendgroup="spy", showlegend=False,
                             line=dict(color="#1f77b4", width=2)), row=2, col=1)

    for r in (1, 2):
        fig.update_yaxes(title_text="净值 (起点=100)", row=r, col=1)
        fig.update_xaxes(title_text="日期", row=r, col=1)

    fig.update_layout(
        title="VIX_ma3/VIX_ma30 比值二值 SPY 策略 · 收益曲线（参数：high_cash, thr=2.00）",
        template="plotly_white", height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    path = f"{OUT_DIR}/vix_ratio_panels.html"
    plot_offline(fig, filename=path, auto_open=False)
    print(f"全样本 2011-2025：策略 +{full_ret:.1f}% ｜ SPY +{full_spy:.1f}%")
    print(f"样本外 2016-2026：策略 +{oos_ret:.1f}% ｜ SPY +{oos_spy:.1f}%")
    print(f"交付：{path}")


if __name__ == "__main__":
    main()
