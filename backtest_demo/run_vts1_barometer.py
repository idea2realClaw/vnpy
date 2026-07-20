"""
run_vts1_barometer.py —— VTS1 轮动策略（基于 13 因子 Volatility Barometer）

策略档位（用户定义）：
    Barometer < 30%     -> 持有 XLP（必需消费，防御）
    30% <= B <= 85%     -> 持有 SPY（S&P500 宽基）
    85% < B <= 90%      -> 持有现金（空仓）
    B > 90%             -> 持有 VXX（波动率 ETN，尾部对冲）

Barometer 来源：复用 vts_barometer13_sim.py 的 13 因子等权合成（旧算法 0-100 口径，
已校验 2020-03-12≈89% / 均值≈50%，与 VTS 官方披露锚点吻合）。该口径下 30/85/90 即
VTS 标准分档阈值。

回测约定（与 run_vts_backtest.py 一致）：t 日收盘算出的 Barometer 信号，决定 t+1 日的
持仓；即第 t 日收益由 baro[t-1] 选定的资产产生（无日内前视）。前导 NaN 期与第 0 日默认 SPY。

窗口：
    “全样本” = Barometer 首个有效日(~2016-01) ~ 最新
    “2020 以来” = 2020-01-01 ~ 最新（含新冠崩盘，检验 >90%→VXX 尾部对冲）

指标：总收益% / 最大回撤% / 年化夏普(无风险=0) / 各资产持有天数占比。
基准：SPY 买入持有（同窗口）。

用法（仓库根目录执行）：
    PYTHONPATH=/Users/zhuxiaodong/Documents/GitRepo/vnpy \
      /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vts1_barometer.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"
W = 1260  # 5 年滚动百分位窗口


# ----------------------------------------------------------------------- 13 因子 Barometer（与 vts_barometer13_sim 一致）
def pct_rank(s, w=W):
    s = pd.Series(s, dtype=float)
    return s.rolling(w, min_periods=w).apply(lambda x: float((x[-1] >= x).mean()), raw=True)


def build_barometer13():
    vix = load_close("VIX"); vix9d = load_close("VIX9D"); vxv = load_close("VXV")
    vvix = load_close("VVIX"); vixm = load_close("VIXM"); spy = load_close("SPY")
    idx = vix.index
    for s in (vix9d, vxv, vvix, vixm, spy):
        idx = idx.intersection(s.index)

    def a(s):
        return s.reindex(idx).astype(float)

    VIX, VIX9D, VXV = a(vix), a(vix9d), a(vxv)
    VVIX, VIXM, SPY = a(vvix), a(vixm), a(spy)
    ret = SPY.pct_change()
    HV20 = ret.rolling(20).std() * np.sqrt(252) * 100
    HV5 = ret.rolling(5).std() * np.sqrt(252) * 100
    sma50 = VIX.rolling(50).mean()

    f = {}
    f["VIX_spot"] = VIX
    f["VIX9D"] = VIX9D
    f["VIX3M_VXV"] = VXV
    f["VIX6M_VIXM"] = VIXM
    f["VVIX"] = VVIX
    f["VIX_VIX3M_ratio"] = VIX / VXV
    f["VIX9D_VIX_ratio"] = VIX9D / VIX
    f["M1_M2_contango"] = VIX - VXV
    f["VX30_VIX_rollyield"] = VIX - VIXM
    f["Simple_VRP"] = VIX - HV20
    w = np.array([1, 2, 3, 4, 5], float); w /= w.sum()
    f["Traders_VRP"] = (VXV - HV5).rolling(5).apply(lambda x: float(np.sum(w * x)), raw=True)
    f["Cash_VIX_Oscillator"] = (VIX - sma50) / sma50
    f["VTS_Cash_VIX_Oscillator"] = VIX - VXV

    pct = {k: pct_rank(v) for k, v in f.items()}
    P = pd.DataFrame(pct)
    return P.mean(axis=1, skipna=True) * 100.0  # 等权（旧算法 0-100）


# ----------------------------------------------------------------------- 档位分配
def allocate(b):
    if np.isnan(b):
        return "SPY"
    if b < 30:
        return "XLP"
    elif b <= 85:
        return "SPY"
    elif b <= 90:
        return "CASH"
    else:
        return "VXX"


def metrics(nav, dates, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    seg = nav[ii[0]:ii[-1] + 1]
    tot = seg[-1] / seg[0] - 1.0
    peak = np.maximum.accumulate(seg)
    mdd = (seg / peak - 1.0).min()
    dr = np.diff(seg) / seg[:-1]
    sharpe = dr.mean() / dr.std(ddof=1) * np.sqrt(252) if dr.std(ddof=1) > 0 else 0.0
    return tot * 100, mdd * 100, sharpe


def main():
    baro = build_barometer13().dropna()  # 首个有效日 ~2016-01
    xlp = load_close("XLP").reindex(baro.index).astype(float)
    spy = load_close("SPY").reindex(baro.index).astype(float)
    vxx = load_close("VXX").reindex(baro.index).astype(float)  # 2018-01-25 起；缺失处按现金回退

    dates = baro.index.to_numpy()
    rxlp = xlp.pct_change().fillna(0.0).to_numpy()
    rspy = spy.pct_change().fillna(0.0).to_numpy()
    rvxx = vxx.pct_change().fillna(0.0).to_numpy()  # 缺数据日(前2018)→0(现金回退)

    N = len(baro)
    alloc = np.empty(N, dtype=object)
    bvals = baro.to_numpy()
    for t in range(N):
        prev = bvals[t - 1] if t >= 1 else np.nan
        alloc[t] = allocate(prev)
    strat_ret = np.where(alloc == "XLP", rxlp,
                np.where(alloc == "SPY", rspy,
                np.where(alloc == "VXX", rvxx, 0.0)))
    strat_ret[0] = 0.0
    nav = np.ones(N)
    for t in range(1, N):
        nav[t] = nav[t - 1] * (1.0 + strat_ret[t])
    nav *= 100.0

    # SPY 买入持有同窗口
    spy_nav = np.ones(N)
    for t in range(1, N):
        spy_nav[t] = spy_nav[t - 1] * (1.0 + rspy[t])
    spy_nav *= 100.0

    # 分配统计
    uniq, cnts = np.unique(alloc, return_counts=True)
    dist = {u: c for u, c in zip(uniq, cnts)}
    total = N

    # 窗口
    d_last = str(pd.Timestamp(dates[-1]).date())
    d_first = str(pd.Timestamp(dates[0]).date())
    WINS = {
        "全样本": (d_first, d_last),
        "2020 以来": ("2020-01-01", d_last),
    }

    rows = []
    for wname, (d0, d1) in WINS.items():
        st, sm, ss = metrics(nav, dates, d0, d1)
        bt, bm, bs = metrics(spy_nav, dates, d0, d1)
        rows.append({
            "窗口": wname,
            "VTS1收益%": round(st, 1), "VTS1回撤%": round(sm, 1), "VTS1夏普": round(ss, 2),
            "SPY收益%": round(bt, 1), "SPY回撤%": round(bm, 1), "SPY夏普": round(bs, 2),
            "超额收益%": round(st - bt, 1),
        })
    df = pd.DataFrame(rows)
    print(f"Barometer 有效区间: {d_first} ~ {d_last} ({N} 交易日)")
    print("分配占比: " + "  ".join(f"{k} {dist.get(k,0)/total*100:.1f}%" for k in ("XLP","SPY","CASH","VXX")))
    print(df.to_string(index=False))
    df.to_csv(f"{OUT}/vts1_barometer.csv", index=False)

    # ---------------- HTML ----------------
    COLORS = {"XLP": "#2ca02c", "SPY": "#1f77b4", "CASH": "#999999", "VXX": "#d62728"}
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.07,
        row_heights=[0.42, 0.30, 0.28],
        subplot_titles=(
            f"VTS1 净值 vs SPY 买入持有（全样本 +{df.iloc[0]['VTS1收益%']}% / SPY +{df.iloc[0]['SPY收益%']}%，2020以来 +{df.iloc[1]['VTS1收益%']}% / SPY +{df.iloc[1]['SPY收益%']}%）",
            "Volatility Barometer（0-100，含分档阈值 30/85/90）",
            "每日持仓（<30 XLP / 30-85 SPY / 85-90 现金 / >90 VXX）",
        ),
    )
    # Row1: NAV
    fig.add_trace(go.Scatter(x=dates, y=nav, name="VTS1 策略", line=dict(color="#d62728", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=spy_nav, name="SPY 买入持有", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.update_yaxes(title_text="净值(起点=100)", row=1, col=1)
    # Row2: Barometer + 分档背景
    fig.add_trace(go.Scatter(x=dates, y=baro.to_numpy(), name="Barometer",
                             line=dict(color="#1f2d5a", width=1.8), showlegend=False), row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#2ca02c", opacity=0.10, line_width=0, row=2, col=1)
    fig.add_hrect(y0=30, y1=85, fillcolor="#1f77b4", opacity=0.08, line_width=0, row=2, col=1)
    fig.add_hrect(y0=85, y1=90, fillcolor="#999999", opacity=0.14, line_width=0, row=2, col=1)
    fig.add_hrect(y0=90, y1=100, fillcolor="#d62728", opacity=0.12, line_width=0, row=2, col=1)
    for thr in (30, 85, 90):
        fig.add_hline(y=thr, line=dict(color="black", width=1, dash="dash"), row=2, col=1)
    fig.update_yaxes(title_text="Barometer", range=[0, 100], row=2, col=1)
    # Row3: 持仓状态带
    for asset, col in COLORS.items():
        y = np.where(alloc == asset, 1.0, np.nan)
        fig.add_trace(go.Scatter(x=dates, y=y, name=asset, mode="lines",
                                 line=dict(color=col, width=9),
                                 showlegend=(asset == "XLP")), row=3, col=1)
    fig.update_yaxes(range=[0.6, 1.4], showticklabels=False, row=3, col=1)

    fig.update_layout(
        title=f"VTS1 轮动策略（Barometer 驱动：<30 XLP / 30-85 SPY / 85-90 现金 / >90 VXX）<br>"
              f"有效区间 {d_first} ~ {d_last}",
        template="plotly_white", height=1040,
        margin=dict(t=80, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="left", x=0),
    )
    path = f"{OUT}/vts1_barometer.html"
    plot_offline(fig, filename=path, auto_open=False)
    print(f"\n交付：{path}\n      {OUT}/vts1_barometer.csv")


if __name__ == "__main__":
    main()
