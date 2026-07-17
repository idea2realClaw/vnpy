"""
run_vix_ratio.py —— 用 VIX_ma3 / VIX_ma30 比值驱动 SPY 二值择时

合成方式（用户指定，2026-07-17）：
    ratio = VIX_ma3 / VIX_ma30
    其中 VIX_ma3 = VIX 的 3 日移动平均（短期），VIX_ma30 = VIX 的 30 日移动平均（长期）
    比值天然融合「短期 vs 长期」波动率结构，且**不依赖任何滚动百分位窗口**，
    因此不会被 2020 年 VIX=80 的极端值污染（不像 pct_rank 窗口会把尾部长期拉高）。

方向：
    high_cash  ratio > thr -> 空仓（危机时短期 VIX 飙升，比值高，应避险）← 直觉方向
    low_cash   ratio < thr -> 空仓（波动率回落极端时反向）

训练：2011-2025 全样本网格搜 (thr, 方向)，目标 = 收益/回撤比 最大。
报告：
    1) 训练所得阈值 + 方向
    2) 2011-2015 用训练参数的实现结果
    3) 2011-2015 网格天花板（过拟合上限，独立在该窗口搜最优）
    4) 2016-2026 样本外（用训练参数，诚实对照）
    5) 危机感知约束版（high_cash 下要求 2020 崩盘期空仓≥50%）+ 收益/回撤比

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ratio.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"


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
    return tot, cagr, mdd, rd


def search(d0, d1, ratio, ret, valid, dates, dt,
           directions=("high_cash", "low_cash"), crisis_mask=None, crash_min=0.0):
    """网格搜 (thr, 方向)，返回 (best, top5)。可选危机约束。"""
    thrs = [round(0.85 + 0.05 * i, 2) for i in range(0, 44)]  # 0.85 .. 3.00
    best = None
    top5 = []
    for direction in directions:
        for thr in thrs:
            nav, pos = nav_for(thr, direction, ratio, ret, valid)
            m = slice_metrics(nav, dates, d0, d1)
            if m is None:
                continue
            tot, cagr, mdd, rd = m
            cash = float(np.mean(pos == 0.0))
            if crisis_mask is not None:
                if crisis_mask.sum() == 0 or float(pos[crisis_mask].mean()) < crash_min:
                    continue
            cand = (rd, thr, direction, tot, cagr, mdd, cash)
            if best is None or rd > best[0]:
                best = cand
            top5.append(cand)
    top5.sort(key=lambda x: -x[0])
    return best, top5[:5]


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

    # ---- 平滑 ----
    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().to_numpy()
    ratio = ma3 / ma30  # 无百分位窗口

    # valid 与 pos 对齐到 ret（长度 N），信号日 dt = dates[:N]
    n = len(ret)
    dt = dates[:n]
    valid = ~np.isnan(ratio[:n]) & (dt >= np.datetime64("2011-01-01"))

    # 危机窗口（信号日口径）
    crash_mask = (dt >= np.datetime64("2020-02-15")) & (dt <= np.datetime64("2020-03-31"))

    print(f"共同交易日: {len(dates)}  ({dates[0].astype('M8[D]')} ~ {dates[-1].astype('M8[D]')})")
    print(f"ratio 范围: {np.nanmin(ratio):.2f} ~ {np.nanmax(ratio):.2f}  "
          f"(2011+ 有效均值 {np.nanmean(ratio[:n][valid]):.3f})")

    # ---- 1) 训练 2011-2025 ----
    tr0, tr1 = "2011-01-01", "2025-12-31"
    best, top5 = search(tr0, tr1, ratio, ret, valid, dates, dt)
    rd_b, thr_b, dir_b, tot_b, cagr_b, mdd_b, cash_b = best
    print("\n=== 训练结果（2011-2025，目标=收益/回撤比）===")
    print(f"  最优：方向={dir_b}  阈值 thr={thr_b:.2f}  "
          f"总收益={tot_b*100:.1f}%  CAGR={cagr_b*100:.1f}%  MDD={mdd_b*100:.1f}%  "
          f"收益/回撤={rd_b:.2f}  空仓={cash_b*100:.1f}%")
    print("  Top5:")
    for c in top5:
        print(f"    dir={c[2]:9s} thr={c[1]:.2f} tot={c[3]*100:.1f}% rd={c[0]:.2f} cash={c[6]*100:.1f}%")

    # ---- 2) 训练参数 → 2011-2015 ----
    nav_tr, pos_tr = nav_for(thr_b, dir_b, ratio, ret, valid)
    early = slice_metrics(nav_tr, dates, "2011-01-01", "2015-12-31")
    oos = slice_metrics(nav_tr, dates, "2016-01-01", "2026-07-15")

    # ---- 3) 2011-2015 网格天花板（过拟合上限）----
    best_e, _ = search("2011-01-01", "2015-12-31", ratio, ret, valid, dates, dt)
    rd_e, thr_e, dir_e, tot_e, cagr_e, mdd_e, cash_e = best_e
    print("\n=== 2011-2015 网格天花板（过拟合上限，独立搜最优）===")
    print(f"  方向={dir_e}  thr={thr_e:.2f}  总收益={tot_e*100:.1f}%  CAGR={cagr_e*100:.1f}%  "
          f"MDD={mdd_e*100:.1f}%  收益/回撤={rd_e:.2f}  空仓={cash_e*100:.1f}%")

    # ---- 4) 危机感知约束（high_cash, 2020 空仓≥50%）----
    best_c, _ = search(tr0, tr1, ratio, ret, valid, dates, dt,
                       directions=("high_cash",), crisis_mask=crash_mask, crash_min=0.50)
    if best_c:
        rd_c, thr_c, dir_c, tot_c, cagr_c, mdd_c, cash_c = best_c
        crash_cash = float(pos_c[crash_mask].mean()) if False else None
        # 重算 crisis 空仓占比
        _, pos_c = nav_for(thr_c, "high_cash", ratio, ret, valid)
        crash_cash = float(pos_c[crash_mask].mean())
        e_c = slice_metrics(nav_for(thr_c, "high_cash", ratio, ret, valid)[0], dates,
                            "2011-01-01", "2015-12-31")
        print("\n=== 危机感知约束版（high_cash, 2020 崩盘空仓≥50%）===")
        print(f"  阈值 thr={thr_c:.2f}  训练期(2011-2025) 总收益={tot_c*100:.1f}%  "
              f"MDD={mdd_c*100:.1f}%  收益/回撤={rd_c:.2f}  空仓={cash_c*100:.1f}%")
        print(f"  2020 崩盘期空仓={crash_cash*100:.1f}%  | 2011-2015 实现："
              f"tot={e_c[0]*100:.1f}% rd={e_c[3]:.2f}")
    else:
        thr_c = None
        print("\n=== 危机感知约束版：无满足 2020 空仓≥50% 的配置 ===")

    # ---- SPY 基准 ----
    spy_nav = np.concatenate([[1.0], np.cumprod(1.0 + ret)])
    spy_train = slice_metrics(spy_nav, dates, tr0, tr1)
    spy_early = slice_metrics(spy_nav, dates, "2011-01-01", "2015-12-31")
    spy_oos = slice_metrics(spy_nav, dates, "2016-01-01", "2026-07-15")
    print(f"\n  SPY 基准：2011-2025 tot={spy_train[0]*100:.1f}% rd={spy_train[3]:.2f} | "
          f"2011-2015 tot={spy_early[0]*100:.1f}% rd={spy_early[3]:.2f} | "
          f"2016-2026 tot={spy_oos[0]*100:.1f}% rd={spy_oos[3]:.2f}")

    # ---- 图表 ----
    def seg_of(nav, d0, d1):
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        seg = nav[ii[0]:ii[-1] + 1]
        return seg / seg[0], dates[ii[0]:ii[-1] + 1]

    train_seg, train_dates = seg_of(nav_tr, tr0, "2026-07-15")
    spy_seg, _ = seg_of(spy_nav, tr0, "2026-07-15")
    crisis_seg = None
    if thr_c is not None:
        nav_c, _ = nav_for(thr_c, "high_cash", ratio, ret, valid)
        crisis_seg, _ = seg_of(nav_c, tr0, "2026-07-15")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_dates, y=spy_seg * 100, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=train_dates, y=train_seg * 100,
                             name=f"VIX比值 (thr={thr_b:.2f}, {dir_b})",
                             line=dict(color="#d62728", width=2.5)))
    if crisis_seg is not None:
        fig.add_trace(go.Scatter(x=train_dates, y=crisis_seg * 100,
                                 name=f"危机感知 (thr={thr_c:.2f}, 2020空仓≥50%)",
                                 line=dict(color="#2ca02c", width=1.8, dash="dot")))
    fig.update_layout(title="VIX_ma3/VIX_ma30 比值二值 SPY 策略净值（2011-2026, 起点=100）",
                      xaxis_title="日期", yaxis_title="净值 (起点=100)",
                      template="plotly_white", height=620,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    path_chart = f"{OUT_DIR}/vix_ratio_chart.html"
    plot_offline(fig, filename=path_chart, auto_open=False)

    # ratio 子图（训练参数阈值线）
    ri_seg = ratio[(dates >= np.datetime64(tr0)) & (dates <= np.datetime64("2026-07-15"))]
    ri_dt = dates[(dates >= np.datetime64(tr0)) & (dates <= np.datetime64("2026-07-15"))]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ri_dt, y=ri_seg, name="ratio=ma3/ma30",
                              line=dict(color="#2ca02c", width=1.3)))
    fig2.add_hline(y=thr_b, line=dict(color="red", dash="dash"),
                   annotation_text=f"训练阈值 {thr_b:.2f} → 空仓", annotation_position="top right")
    if thr_c is not None:
        fig2.add_hline(y=thr_c, line=dict(color="green", dash="dash"),
                       annotation_text=f"危机阈值 {thr_c:.2f}", annotation_position="bottom right")
    fig2.update_layout(title="合成 ratio 与阈值线", xaxis_title="日期",
                       yaxis_title="VIX_ma3 / VIX_ma30", template="plotly_white", height=380)
    path_ind = f"{OUT_DIR}/vix_ratio_indicator.html"
    plot_offline(fig2, filename=path_ind, auto_open=False)

    # ---- CSV ----
    rows = [
        ("SPY 买入持有 (基准)", *[round(x, 4) for x in spy_train], 0.0),
        ("训练最优 (2011-2025)", tot_b, cagr_b, mdd_b, rd_b, cash_b),
        ("训练参数→2011-2015", *early, float(np.mean(pos_tr == 0.0))),
        ("训练参数→2016-2026 OOS", *oos, float(np.mean(pos_tr == 0.0))),
        ("2011-2015 网格天花板", tot_e, cagr_e, mdd_e, rd_e, cash_e),
    ]
    if thr_c is not None:
        rows.append((f"危机感知 (thr={thr_c:.2f},2020空仓≥50%)", tot_c, cagr_c, mdd_c, rd_c, cash_c))
    df = pd.DataFrame(rows, columns=["策略", "总收益", "CAGR", "最大回撤", "收益/回撤", "空仓占比"])
    df.to_csv(f"{OUT_DIR}/vix_ratio_summary.csv", index=False)

    # ---- 关键结论 ----
    print("\n=== 关键结论 ===")
    print(f"合成方式：ratio = VIX_ma3 / VIX_ma30（无百分位窗口，抗 2020 污染）")
    print(f"训练所得：方向={dir_b}  阈值 thr={thr_b:.2f}  训练期空仓={cash_b*100:.1f}%")
    print(f"2011-2015 最佳实现（训练参数）：总收益={early[0]*100:.1f}%，收益/回撤={early[3]:.2f}"
          f"（SPY 基准={spy_early[0]*100:.1}%，ret/DD={spy_early[3]:.2f}）")
    if thr_c is not None:
        print(f"危机感知版：thr={thr_c:.2f}，2020 崩盘空仓={crash_cash*100:.1f}%")
    print(f"\n交付：{path_chart}\n      {path_ind}\n      {OUT_DIR}/vix_ratio_summary.csv")


if __name__ == "__main__":
    main()
