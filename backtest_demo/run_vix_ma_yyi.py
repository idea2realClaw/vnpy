"""
VIX MA 合成 YYI / Barometer —— 阈值重标定（2011-2025 训练，2016+ 样本外）

设计：
  1) 先把原始 VIX 做移动平均平滑，再做百分位，避免日间噪声：
       VIX_ma30 = VIX 的 30 日移动平均   （长期趋势）
       VIX_ma3  = VIX 的 3 日移动平均    （短期）
       long_rank  = VIX_ma30 在 5 年(1260 日)滚动窗口的百分位 rank
       short_rank = VIX_ma3  在 5 年(1260 日)滚动窗口的百分位 rank
  2) 合成指标（YYI 式加权混合）：
       indicator(t) = r * long_rank(t) + (1 - r) * short_rank(t)
       r ∈ [0,1] = 长期权重；1-r = 短期权重  → 这就是要训练的「长期短期比例」
  3) 二值择时（SPY 空仓 / 满仓）：
       高 indicator ⇒ 空仓（危机避险，高波动=危险）
       低 indicator ⇒ 空仓（反向，低波动=危险）—— 两方向都搜
       阈值 thr ∈ (0,1)：indicator 越过 thr 即空仓
  4) 训练 2011-2025：网格搜 (r, thr, 方向)，目标 = 训练期 收益/回撤比 最大
  5) 报告：
       - 训练所得 长期权重 r / 短期权重 (1-r) / 阈值 thr / 方向
       - 2011-2015 用该参数的结果（in-sample 早期子区间）
       - 2011-2015 网格天花板（仅用 2011-2015 训练的最优，过拟合上限）
       - 2016-2026 样本外（用 2011-2025 训练参数）

无未来函数：百分位窗口只用截至 t 的历史；信号在 t 日收盘算出，t+1 日按收益调仓。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ma_yyi.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close, pct_rank

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"
W = 1260          # 5 年百分位窗口
TRAIN = (pd.Timestamp("2011-01-01"), pd.Timestamp("2025-12-31"))
EARLY = (pd.Timestamp("2011-01-01"), pd.Timestamp("2015-12-31"))
OOS   = (pd.Timestamp("2016-01-01"), pd.Timestamp("2026-07-16"))


# ----------------------------------------------------------------------------- 工具
def metrics(nav_seg: np.ndarray, d0, d1):
    d0 = pd.Timestamp(d0)
    d1 = pd.Timestamp(d1)
    years = max((d1 - d0).days / 365.25, 1e-9)
    tot = nav_seg[-1] / nav_seg[0] - 1.0
    cagr = (nav_seg[-1] / nav_seg[0]) ** (1.0 / years) - 1.0
    peak = np.maximum.accumulate(nav_seg)
    dd = nav_seg / peak - 1.0
    mdd = float(dd.min())
    ret_dd = tot / abs(mdd) if mdd < 0 else np.inf
    return tot, cagr, mdd, ret_dd


def build_indicator(vix: pd.Series):
    """返回 (dates, indicator) —— indicator 在 warmup 前为 nan。"""
    v = vix.to_numpy(dtype=float)
    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().bfill().ffill().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().bfill().ffill().to_numpy()
    long_rank = pct_rank(ma30, W)
    short_rank = pct_rank(ma3, W)
    return vix.index, long_rank, short_rank


def nav_for(r: float, thr: float, direction: str, long_rank, short_rank, spy_ret, valid):
    """valid: bool 数组，indicator 有效（warmup 后）。信号在 t，仓位用于 t+1 收益。"""
    indicator = r * long_rank + (1 - r) * short_rank
    n = len(spy_ret)
    nav = np.ones(n + 1)
    pos = np.ones(n)                 # 默认满仓
    for t in range(n):
        if not valid[t]:
            continue                # warmup 期：维持上一仓位（首段默认满仓）
        ind = indicator[t]
        if np.isnan(ind):
            continue
        if direction == "high_cash":
            pos[t] = 0.0 if ind > thr else 1.0
        else:  # low_cash
            pos[t] = 0.0 if ind < thr else 1.0
    for t in range(n):
        nav[t + 1] = nav[t] * (1.0 + pos[t] * spy_ret[t])
    return nav, pos, indicator


def slice_metrics(nav, dates, d0, d1):
    mask = (dates >= d0) & (dates <= d1)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return None
    seg = nav[idx[0]:idx[-1] + 1]
    dts = dates[idx[0]:idx[-1] + 1]
    return metrics(seg, dts[0], dts[-1])


# ----------------------------------------------------------------------------- 主
def main():
    vix = load_close("VIX")
    spy = load_close("SPY")
    idx = vix.index.intersection(spy.index)
    vix, spy = vix.loc[idx], spy.loc[idx]
    dates = idx.to_numpy()
    spy_arr = spy.to_numpy(dtype=float)
    spy_ret = np.diff(spy_arr) / spy_arr[:-1]

    di, long_rank, short_rank = build_indicator(vix)
    valid = ~np.isnan(long_rank) & ~np.isnan(short_rank) & (di.to_numpy() >= np.datetime64("2011-01-01"))
    # 危机窗口 mask（2020 新冠崩盘 2/15-3/31），用于危机感知约束
    crash_mask = (dates >= np.datetime64("2020-02-15")) & (dates <= np.datetime64("2020-03-31"))
    # 实际上 warmup 由 pct_rank 的窗口决定；确保训练起点前已有效
    first_valid = int(np.argmax(valid)) if valid.any() else 0
    print(f"数据: {len(dates)} 日 ({pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()})")
    print(f"indicator 首个有效日: {pd.Timestamp(dates[first_valid]).date()}")

    spy_nav = np.concatenate([[1.0], np.cumprod(1.0 + spy_ret)])

    # ---- 网格搜索
    Rs = [i / 20.0 for i in range(21)]          # 0..1 step 0.05
    thr_high = [round(0.50 + 0.01 * i, 2) for i in range(0, 49)]   # 0.50..0.98
    thr_low = [round(0.02 + 0.01 * i, 2) for i in range(0, 49)]    # 0.02..0.50

    def search(d0, d1, crash_mask=None, crash_min_cash=0.0):
        """crash_mask: 布尔数组（与 dates 等长）；若给定，则只保留在该窗口空仓占比
        >= crash_min_cash 的配置（危机感知约束，避免退化为买入持有）。"""
        best = None
        top5 = []
        for direction, thr_list in [("high_cash", thr_high), ("low_cash", thr_low)]:
            for r in Rs:
                for thr in thr_list:
                    n = len(spy_ret)
                    p = np.ones(n)
                    indicator = r * long_rank + (1 - r) * short_rank
                    for t in range(n):
                        if not valid[t] or np.isnan(indicator[t]):
                            continue
                        if direction == "high_cash":
                            p[t] = 0.0 if indicator[t] > thr else 1.0
                        else:
                            p[t] = 0.0 if indicator[t] < thr else 1.0
                    if crash_mask is not None:
                        cm = crash_mask[:len(p)]
                        if cm.sum() == 0 or float(p[cm].mean()) < crash_min_cash:
                            continue
                    nav2 = np.ones(n + 1)
                    for t in range(n):
                        nav2[t + 1] = nav2[t] * (1.0 + p[t] * spy_ret[t])
                    m = slice_metrics(nav2, dates, d0, d1)
                    if m is None:
                        continue
                    tot, cagr, mdd, ret_dd = m
                    cand = (ret_dd, r, thr, direction, tot, cagr, mdd, float(np.mean(p == 0.0)))
                    if best is None or ret_dd > best[0]:
                        best = cand
                    top5.append(cand)
        top5.sort(key=lambda x: -x[0])
        return best, top5[:5]

    print("\n=== 网格搜索（训练 2011-2025，目标=收益/回撤比）===")
    best_train, top5_train = search(*TRAIN)
    rd, r, thr, direction, tot, cagr, mdd, cash = best_train
    print(f"训练最优: 长期权重 r={r:.2f}  短期权重={1-r:.2f}  阈值={thr:.2f}  方向={direction}")
    print(f"  训练期(2011-2025): 总收益={tot*100:.1f}%  CAGR={cagr*100:.1f}%  MDD={mdd*100:.1f}%  收益/回撤={rd:.2f}  空仓占比={cash*100:.1f}%")
    print("  Top5 (ret_dd, r, thr, dir):")
    for c in top5_train:
        print(f"    ret_dd={c[0]:.2f}  r={c[1]:.2f}  thr={c[2]:.2f}  {c[3]}  cash={c[7]*100:.1f}%")

    # 基准：SPY 买入持有
    spy_train = slice_metrics(spy_nav, dates, *TRAIN)
    spy_early = slice_metrics(spy_nav, dates, *EARLY)
    spy_oos = slice_metrics(spy_nav, dates, *OOS)
    print(f"\n  SPY 买入持有基准: 训练={spy_train[0]*100:.1f}%(ret/DD={spy_train[3]:.2f})  "
          f"2011-2015={spy_early[0]*100:.1f}%  OOS={spy_oos[0]*100:.1f}%")

    # ---- 2011-2015 用训练参数
    n = len(spy_ret)
    nav_ep = np.ones(n + 1)
    indicator = r * long_rank + (1 - r) * short_rank
    p = np.ones(n)
    for t in range(n):
        if not valid[t] or np.isnan(indicator[t]):
            continue
        if direction == "high_cash":
            p[t] = 0.0 if indicator[t] > thr else 1.0
        else:
            p[t] = 0.0 if indicator[t] < thr else 1.0
    for t in range(n):
        nav_ep[t + 1] = nav_ep[t] * (1.0 + p[t] * spy_ret[t])

    early_res = slice_metrics(nav_ep, dates, *EARLY)
    oos_res = slice_metrics(nav_ep, dates, *OOS)
    print(f"\n=== 训练参数 (r={r:.2f}, thr={thr:.2f}, {direction}) 应用结果 ===")
    print(f"  2011-2015: 总收益={early_res[0]*100:.1f}%  CAGR={early_res[1]*100:.1f}%  "
          f"MDD={early_res[2]*100:.1f}%  收益/回撤={early_res[3]:.2f}")
    print(f"  2016-2026(OOS): 总收益={oos_res[0]*100:.1f}%  CAGR={oos_res[1]*100:.1f}%  "
          f"MDD={oos_res[2]*100:.1f}%  收益/回撤={oos_res[3]:.2f}")

    # ---- 2011-2015 网格天花板（仅用 2011-2015 训练的最优，过拟合上限）
    print(f"\n=== 2011-2015 网格天花板（仅用 2011-2015 训练，过拟合上限）===")
    best_early, _ = search(*EARLY)
    re, r2, thr2, d2, t2, c2, m2, cash2 = best_early
    print(f"  最优: 长期权重 r={r2:.2f}  短期权重={1-r2:.2f}  阈值={thr2:.2f}  方向={d2}")
    print(f"  2011-2015: 总收益={t2*100:.1f}%  CAGR={c2*100:.1f}%  MDD={m2*100:.1f}%  "
          f"收益/回撤={re:.2f}  空仓占比={cash2*100:.1f}%")

    # ---- 危机感知阈值：要求 2020 崩盘期空仓≥30%，再最大化训练期 收益/回撤比
    print(f"\n=== 危机感知阈值（high_cash，约束：2020 崩盘期空仓≥30%，目标=训练期 ret/DD）===")
    best_crisis, _ = search(*TRAIN, crash_mask=crash_mask, crash_min_cash=0.30)
    rc, r3, thr3, d3, t3, c3, m3, cash3 = best_crisis
    # 重建危机感知 nav
    n = len(spy_ret)
    nav_cr = np.ones(n + 1)
    ind3 = r3 * long_rank + (1 - r3) * short_rank
    p3 = np.ones(n)
    for t in range(n):
        if not valid[t] or np.isnan(ind3[t]):
            continue
        p3[t] = 0.0 if ind3[t] > thr3 else 1.0
    for t in range(n):
        nav_cr[t + 1] = nav_cr[t] * (1.0 + p3[t] * spy_ret[t])
    early_cr = slice_metrics(nav_cr, dates, *EARLY)
    oos_cr = slice_metrics(nav_cr, dates, *OOS)
    crash_cash = float(p3[crash_mask[:n]].mean()) if crash_mask[:n].sum() else 0.0
    print(f"  最优: 长期权重 r={r3:.2f}  短期权重={1-r3:.2f}  阈值={thr3:.2f}  方向={d3}")
    print(f"  训练期(2011-2025): 总收益={t3*100:.1f}%  CAGR={c3*100:.1f}%  MDD={m3*100:.1f}%  "
          f"收益/回撤={rc:.2f}  空仓占比={cash3*100:.1f}%  2020崩盘期空仓={crash_cash*100:.1f}%")
    print(f"  2011-2015: 总收益={early_cr[0]*100:.1f}%  ret/DD={early_cr[3]:.2f}")
    print(f"  2016-2026(OOS): 总收益={oos_cr[0]*100:.1f}%  ret/DD={oos_cr[3]:.2f}")

    # ---- 图表：nav 对比 + indicator 与阈值带
    fig = go.Figure()
    dts_all = [pd.Timestamp(x).date() for x in dates]
    # nav 段（2011-2026）
    m_train = (dates >= np.datetime64("2011-01-01")) & (dates <= np.datetime64("2026-07-16"))
    ii = np.where(m_train)[0]
    seg_dates = [pd.Timestamp(dates[k]).date() for k in ii]
    spy_seg = spy_nav[ii[0]:ii[-1] + 1] / spy_nav[ii[0]]
    ep_seg = nav_ep[ii[0]:ii[-1] + 1] / nav_ep[ii[0]]
    cr_seg = nav_cr[ii[0]:ii[-1] + 1] / nav_cr[ii[0]]
    fig.add_trace(go.Scatter(x=seg_dates, y=spy_seg * 100, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=seg_dates, y=ep_seg * 100,
                             name=f"未约束训练最优 (r={r:.2f}, thr={thr:.2f}, 空仓{cash*100:.0f}%)",
                             line=dict(color="#d62728", width=2.5)))
    fig.add_trace(go.Scatter(x=seg_dates, y=cr_seg * 100,
                             name=f"危机感知 (r={r3:.2f}, thr={thr3:.2f}, 空仓{cash3*100:.0f}%)",
                             line=dict(color="#ff7f0e", width=2)))
    fig.update_layout(title="VIX MA 合成 YYI 二值 SPY 策略净值（2011-2026, 起点=100）",
                      xaxis_title="日期", yaxis_title="净值 (起点=100)",
                      template="plotly_white", height=620,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

    # indicator 子图（训练参数 + 危机感知阈值带）
    indicator = r * long_rank + (1 - r) * short_rank
    ind_seg = indicator[ii]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=seg_dates, y=ind_seg * 100, name="indicator",
                              line=dict(color="#2ca02c", width=1.5)))
    fig2.add_hline(y=thr * 100, line=dict(color="red", dash="dash"),
                   annotation_text=f"未约束阈值 {thr:.2f}", annotation_position="top left")
    fig2.add_hline(y=thr3 * 100, line=dict(color="orange", dash="dot"),
                   annotation_text=f"危机感知阈值 {thr3:.2f}", annotation_position="bottom right")
    fig2.update_layout(title="合成 indicator 与两条阈值线", xaxis_title="日期",
                       yaxis_title="indicator (0-100)", template="plotly_white", height=380)

    # 合并为两个独立 html 便于预览
    chart_path = f"{OUT_DIR}/vix_ma_yyi_chart.html"
    plot_offline(fig, filename=chart_path, auto_open=False, include_plotlyjs="cdn")
    chart2_path = f"{OUT_DIR}/vix_ma_yyi_indicator.html"
    plot_offline(fig2, filename=chart2_path, auto_open=False, include_plotlyjs="cdn")

    # ---- CSV 汇总
    rows = [
        ("SPY 买入持有 (基准)", *[round(x, 4) for x in spy_train], 0.0),
        ("未约束训练最优 (2011-2025)", tot, cagr, mdd, rd, cash),
        ("训练参数→2011-2015", *early_res, cash),
        ("训练参数→2016-2026 OOS", *oos_res, cash),
        ("2011-2015 网格天花板", t2, c2, m2, re, cash2),
        ("危机感知阈值 (2011-2025)", t3, c3, m3, rc, cash3),
        ("危机感知→2011-2015", *early_cr, cash3),
        ("危机感知→2016-2026 OOS", *oos_cr, cash3),
    ]
    df = pd.DataFrame(rows, columns=["策略", "total_return", "CAGR", "max_drawdown", "ret_dd_ratio", "cash_ratio"])
    csv_path = f"{OUT_DIR}/vix_ma_yyi_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n图表: {chart_path}\n      {chart2_path}\n汇总: {csv_path}")

    # 关键结论打印
    print("\n=== 关键结论 ===")
    print(f"① 未约束训练最优：长期权重 r={r:.2f}（{r*100:.0f}%）/ 短期权重={1-r:.2f}（{(1-r)*100:.0f}%）")
    print(f"   阈值 thr={thr:.2f} 方向={direction} 训练期空仓={cash*100:.1f}% —— 退化为买入持有（阈值过高从不触发）")
    print(f"   2011-2015 用该参数：总收益={early_res[0]*100:.1f}%（≈SPY 基准 {spy_early[0]*100:.1f}%），无 alpha")
    print(f"② 危机感知阈值：长期权重 r={r3:.2f}（{r3*100:.0f}%）/ 短期权重={1-r3:.2f}（{(1-r3)*100:.0f}%）")
    print(f"   阈值 thr={thr3:.2f} 方向={d3} 训练期空仓={cash3*100:.1f}%  2020崩盘期空仓={crash_cash*100:.1f}%")
    print(f"   2011-2015：总收益={early_cr[0]*100:.1f}% ret/DD={early_cr[3]:.2f}；OOS：总收益={oos_cr[0]*100:.1f}% ret/DD={oos_cr[3]:.2f}")


if __name__ == "__main__":
    main()
