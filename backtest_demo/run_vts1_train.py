"""
run_vts1_train.py —— 用 2016-2021 训练期标定 VTS1 轮动阈值，2022-2026 样本外验证。

策略结构（4 档，3 道阈值）：
    Barometer < a        -> XLP（防御）
    a <= B < b           -> SPY（宽基）
    b <= B < c           -> 现金（空仓）
    B >= c               -> VXX（尾部对冲）

训练：在 2016-01-06 ~ 2021-12-31 上网格搜索 (a, b, c)，目标 = 年化夏普最高。
样本外：固定最优阈值，在 2022-01-01 ~ 最新 回测，对比 SPY 买入持有。
对照：用户原固定阈值 30/85/90 在两窗口的表现，以及 Top5 阈值表。

Barometer：13 因子等权旧算法(0-100)，已校验锚点。信号 t-1→持仓 t（无日内前视）。

用法（仓库根目录执行）：
    PYTHONPATH=/Users/zhuxiaodong/Documents/GitRepo/vnpy \
      /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vts1_train.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts1_barometer import build_barometer13, load_close

OUT = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"
TRAIN = ("2016-01-06", "2021-12-31")
TEST = ("2022-01-01", "2026-07-17")


def nav_for(a, b, c, baro, rxlp, rspy, rvxx):
    N = len(baro)
    bv = baro.to_numpy()
    strat_ret = np.empty(N)
    for t in range(N):
        prev = bv[t - 1] if t >= 1 else np.nan
        if np.isnan(prev) or prev < a:
            r = rxlp[t]
        elif prev < b:
            r = rspy[t]
        elif prev < c:
            r = 0.0
        else:
            r = rvxx[t]
        strat_ret[t] = r
    strat_ret[0] = 0.0
    nav = np.ones(N)
    for t in range(1, N):
        nav[t] = nav[t - 1] * (1.0 + strat_ret[t])
    return nav * 100.0


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
    baro = build_barometer13().dropna()
    xlp = load_close("XLP").reindex(baro.index).astype(float)
    spy = load_close("SPY").reindex(baro.index).astype(float)
    vxx = load_close("VXX").reindex(baro.index).astype(float)
    dates = baro.index.to_numpy()
    rxlp = xlp.pct_change().fillna(0.0).to_numpy()
    rspy = spy.pct_change().fillna(0.0).to_numpy()
    rvxx = vxx.pct_change().fillna(0.0).to_numpy()

    spy_nav = np.ones(len(baro))
    for t in range(1, len(baro)):
        spy_nav[t] = spy_nav[t - 1] * (1.0 + rspy[t])
    spy_nav *= 100.0

    # ---------- 网格搜索 ----------
    A = [10, 15, 20, 25, 30, 35, 40]
    B = [50, 55, 60, 65, 70, 75, 80, 85]
    C = [85, 88, 90, 92, 94, 96]
    grid = []
    for a in A:
        for b in B:
            if b <= a:
                continue
            for c in C:
                if c <= b:
                    continue
                nav = nav_for(a, b, c, baro, rxlp, rspy, rvxx)
                tr, tm, ts = metrics(nav, dates, *TRAIN)
                grid.append({"a": a, "b": b, "c": c,
                             "训练收益%": round(tr, 1), "训练回撤%": round(tm, 1),
                             "训练夏普": round(ts, 3)})
    gdf = pd.DataFrame(grid).sort_values("训练夏普", ascending=False).reset_index(drop=True)
    best = gdf.iloc[0]
    a0, b0, c0 = int(best["a"]), int(best["b"]), int(best["c"])

    # ---------- 样本外 + 对照 ----------
    nav_best = nav_for(a0, b0, c0, baro, rxlp, rspy, rvxx)
    bt, bm, bs = metrics(nav_best, dates, *TRAIN)
    ot, om, os_ = metrics(nav_best, dates, *TEST)
    st, sm, ss = metrics(spy_nav, dates, *TRAIN)
    ost, osm, oss = metrics(spy_nav, dates, *TEST)

    # 固定 30/85/90
    nav_fix = nav_for(30, 85, 90, baro, rxlp, rspy, rvxx)
    ft, fm, fs = metrics(nav_fix, dates, *TRAIN)
    fot, fom, fos = metrics(nav_fix, dates, *TEST)

    print(f"=== 网格搜索(训练期 {TRAIN[0]}~{TRAIN[1]}, 目标=夏普最高) ===")
    print(f"最优阈值: a(XLP|SPY)={a0}  b(SPY|现金)={b0}  c(现金|VXX)={c0}")
    print("\nTop5:")
    print(gdf.head(5).to_string(index=False))
    print(f"\n=== 最优阈值表现 ===")
    print(f"训练期 2016-2021 : VTS1 +{bt:.1f}% / 回撤 {bm:.1f}% / 夏普 {bs:.2f}  || SPY +{st:.1f}% / {sm:.1f}% / {ss:.2f}")
    print(f"样本外 2022-2026 : VTS1 +{ot:.1f}% / 回撤 {om:.1f}% / 夏普 {os_:.2f}  || SPY +{ost:.1f}% / {osm:.1f}% / {oss:.2f}")
    print(f"\n=== 固定 30/85/90 对照 ===")
    print(f"训练期 2016-2021 : VTS1 +{ft:.1f}% / 回撤 {fm:.1f}% / 夏普 {fs:.2f}")
    print(f"样本外 2022-2026 : VTS1 +{fot:.1f}% / 回撤 {fom:.1f}% / 夏普 {fos:.2f}")

    # 分配占比(样本外, 看是否真去对冲)
    def alloc_dist(a, b, c):
        N = len(baro); bv = baro.to_numpy(); cnt = {"XLP": 0, "SPY": 0, "CASH": 0, "VXX": 0}
        for t in range(N):
            prev = bv[t - 1] if t >= 1 else np.nan
            if np.isnan(prev) or prev < a:
                cnt["XLP"] += 1
            elif prev < b:
                cnt["SPY"] += 1
            elif prev < c:
                cnt["CASH"] += 1
            else:
                cnt["VXX"] += 1
        return {k: v / N * 100 for k, v in cnt.items()}
    print("\n样本外持仓占比(最优):", {k: f"{v:.1f}%" for k, v in alloc_dist(a0, b0, c0).items()})

    # ---------- 保存 ----------
    gdf.to_csv(f"{OUT}/vts1_train_grid.csv", index=False)
    summ = pd.DataFrame([
        {"窗口": "训练 2016-2021", "方案": f"最优(a={a0},b={b0},c={c0})",
         "收益%": round(bt, 1), "回撤%": round(bm, 1), "夏普": round(bs, 2),
         "SPY收益%": round(st, 1), "SPY回撤%": round(sm, 1), "SPY夏普": round(ss, 2)},
        {"窗口": "样本外 2022-2026", "方案": f"最优(a={a0},b={b0},c={c0})",
         "收益%": round(ot, 1), "回撤%": round(om, 1), "夏普": round(os_, 2),
         "SPY收益%": round(ost, 1), "SPY回撤%": round(osm, 1), "SPY夏普": round(oss, 2)},
        {"窗口": "训练 2016-2021", "方案": "固定30/85/90",
         "收益%": round(ft, 1), "回撤%": round(fm, 1), "夏普": round(fs, 2),
         "SPY收益%": round(st, 1), "SPY回撤%": round(sm, 1), "SPY夏普": round(ss, 2)},
        {"窗口": "样本外 2022-2026", "方案": "固定30/85/90",
         "收益%": round(fot, 1), "回撤%": round(fom, 1), "夏普": round(fos, 2),
         "SPY收益%": round(ost, 1), "SPY回撤%": round(osm, 1), "SPY夏普": round(oss, 2)},
    ])
    summ.to_csv(f"{OUT}/vts1_train_summary.csv", index=False)

    # ---------- HTML ----------
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.07,
                        row_heights=[0.36, 0.36, 0.28],
                        subplot_titles=(
        f"训练期 2016-2021 ｜ 最优 VTS1 +{bt:.1f}% / SPY +{st:.1f}% (夏普 {bs:.2f} vs {ss:.2f})",
        f"样本外 2022-2026 ｜ 最优 VTS1 +{ot:.1f}% / SPY +{ost:.1f}% (夏普 {os_:.2f} vs {oss:.2f})",
        f"Barometer 分档(训练出最优阈值 a={a0}/b={b0}/c={c0})",
    ))
    di_tr = (dates >= np.datetime64(TRAIN[0])) & (dates <= np.datetime64(TRAIN[1]))
    di_te = (dates >= np.datetime64(TEST[0])) & (dates <= np.datetime64(TEST[1]))
    # Row1 train
    fig.add_trace(go.Scatter(x=dates[di_tr], y=nav_best[di_tr], name="最优 VTS1", line=dict(color="#d62728", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates[di_tr], y=spy_nav[di_tr], name="SPY 买入持有", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.update_yaxes(title_text="净值(起点=100)", row=1, col=1)
    # Row2 test
    fig.add_trace(go.Scatter(x=dates[di_te], y=nav_best[di_te], name="最优 VTS1", showlegend=False, line=dict(color="#d62728", width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates[di_te], y=spy_nav[di_te], name="SPY 买入持有", showlegend=False, line=dict(color="#1f77b4", width=2)), row=2, col=1)
    fig.update_yaxes(title_text="净值(起点=100)", row=2, col=1)
    # Row3 barometer bands
    fig.add_trace(go.Scatter(x=dates, y=baro.to_numpy(), name="Barometer", showlegend=False, line=dict(color="#1f2d5a", width=1.6)), row=3, col=1)
    fig.add_hrect(y0=0, y1=a0, fillcolor="#2ca02c", opacity=0.10, line_width=0, row=3, col=1,
                  annotation_text=f"<{a0} XLP", annotation_position="top left")
    fig.add_hrect(y0=a0, y1=b0, fillcolor="#1f77b4", opacity=0.08, line_width=0, row=3, col=1,
                  annotation_text=f"{a0}-{b0} SPY", annotation_position="top left")
    fig.add_hrect(y0=b0, y1=c0, fillcolor="#999999", opacity=0.14, line_width=0, row=3, col=1,
                  annotation_text=f"{b0}-{c0} 现金", annotation_position="top left")
    fig.add_hrect(y0=c0, y1=100, fillcolor="#d62728", opacity=0.12, line_width=0, row=3, col=1,
                  annotation_text=f">{c0} VXX", annotation_position="top left")
    fig.update_yaxes(title_text="Barometer", range=[0, 100], row=3, col=1)

    fig.update_layout(
        title=f"VTS1 阈值训练(2016-2021)与样本外(2022-2026)<br>最优: a={a0}(XLP|SPY) / b={b0}(SPY|现金) / c={c0}(现金|VXX)",
        template="plotly_white", height=1080,
        margin=dict(t=80, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="left", x=0),
    )
    path = f"{OUT}/vts1_train.html"
    plot_offline(fig, filename=path, auto_open=False)
    print(f"\n交付：{path}\n      {OUT}/vts1_train_grid.csv\n      {OUT}/vts1_train_summary.csv")


if __name__ == "__main__":
    main()
