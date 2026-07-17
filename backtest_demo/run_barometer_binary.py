"""Barometer 阈值重标定 + 样本外二值 SPY 回测（空仓/满仓）
=================================================

目标：解决前述「真实 Barometer 分布集中在 20-66 中带、套用 VTS 固定阈值(<20/>85→Cash)
几乎不去风险、无杠杆下跑输买入持有」的问题。

方法（walk-forward / 样本外）：
  训练：2011-2015 在真实 Barometer 上扫描单一阈值 T，找「最优去风险阈值」。
        规则 = 二值 SPY：Barometer 高 -> 空仓，低 -> 满仓 SPY（也试反方向，让数据选）。
        目标 = 训练期 收益/回撤比（risk-adjusted）最大。
  样本外：用训练得到的最优阈值，回测 2016-01-01 ~ 今。
  对比：SPY 买入持有、VTS 固定阈值(>=85→Cash / <20→Cash)、以及 ±5/±10 阈值敏感性。

信号口径：t 日收盘 Barometer -> 决定 t 日收盘到 t+1 日收盘这段收益是否持仓（无未来函数）。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_barometer_binary.py
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from backtest_demo.vts_barometer_formula import build_barometer

DB_PATH = "/Users/zhuxiaodong/.vntrader/database.db"
OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"

TRAIN_START = pd.Timestamp("2011-01-01")
TRAIN_END = pd.Timestamp("2015-12-31")
TEST_START = pd.Timestamp("2016-01-01")
TEST_END = pd.Timestamp("2026-07-15")
GRID = list(range(35, 86, 1))   # 扫描阈值 35..85（Barometer 集中在 20-66，留余量到危机区）


# ----------------------------------------------------------------------------- 数据
def load_spy() -> pd.Series:
    con = sqlite3.connect(DB_PATH)
    d = pd.read_sql(
        "SELECT datetime, close_price FROM dbbardata WHERE symbol='SPY' AND exchange='SMART' ORDER BY datetime",
        con,
    )
    con.close()
    d["dt"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None).dt.normalize()
    return d.set_index("dt")["close_price"].astype(float)


def align(baro: pd.Series, spy: pd.Series):
    """对齐 Barometer 与 SPY 到共同交易日；返回 (dates, baro_arr, spy_ret_arr)。"""
    common = baro.index.intersection(spy.index)
    b = baro.reindex(common)
    p = spy.reindex(common)
    rets = p.diff().to_numpy() / p.shift(1).to_numpy()
    rets = rets[1:]                      # 长度 = 共同日数 - 1
    return common, b.to_numpy(), rets


# ----------------------------------------------------------------------------- 组合
def make_nav(baro_arr: np.ndarray, rets: np.ndarray, rule) -> np.ndarray:
    """rule(b) -> True 表示满仓 SPY；NaN/缺失 -> 满仓（中性=持仓）。
    返回 nav（长度 = len(rets)+1，nav[0]=1）。"""
    n = len(rets)
    nav = np.ones(n + 1)
    for t in range(1, n + 1):
        b = baro_arr[t - 1]
        hold = True if (np.isnan(b) or rule(b)) else False
        nav[t] = nav[t - 1] * (1.0 + (rets[t - 1] if hold else 0.0))
    return nav


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def stats(nav_seg: np.ndarray, years: float):
    tr = nav_seg[-1] / nav_seg[0] - 1.0
    cagr = (nav_seg[-1] / nav_seg[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    mdd = max_drawdown(nav_seg)
    rdd = tr / abs(mdd) if mdd != 0 else 0.0
    return round(tr * 100, 2), round(cagr * 100, 2), round(mdd * 100, 2), round(rdd, 2)


def make_rule(direction: str, T: float):
    if direction == "high_cash":        # 高 Barometer = 高波动 = 空仓
        return lambda b: (b < T)
    else:                               # low_cash：低 Barometer = 空仓
        return lambda b: (b > T)


# ----------------------------------------------------------------------------- 主流程
def run():
    baro, _, _ = build_barometer()      # 真实重建 Barometer（含 VVIX+VIX9D，16 分量）
    spy = load_spy()
    dates, baro_arr, rets = align(baro, spy)
    print(f"共同交易日: {len(dates)}  ({dates[0].date()} ~ {dates[-1].date()})")
    print(f"Barometer 有效起点(非NaN): {baro.dropna().index[0].date()}  "
          f"(2011 为 warmup，填中性=持仓)")

    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    test_mask = (dates >= TEST_START) & (dates <= pd.Timestamp(TEST_END))
    yrs_train = (dates[train_mask].max() - dates[train_mask].min()).days / 365.25
    yrs_test = (dates[test_mask].max() - dates[test_mask].min()).days / 365.25
    print(f"训练窗 {TRAIN_START.date()}~{TRAIN_END.date()} ({yrs_train:.1f}年) | "
          f"样本外 {TEST_START.date()}~{TEST_END} ({yrs_test:.1f}年)\n")

    # ---- 训练：扫描方向 × 阈值，目标 = 训练期收益/回撤比最大 ----
    best = None
    grid_rows = []
    for direction in ["high_cash", "low_cash"]:
        for T in GRID:
            rule = make_rule(direction, T)
            nav = make_nav(baro_arr, rets, rule)
            seg = nav[train_mask.values if hasattr(train_mask, "values") else train_mask]
            tr, cagr, mdd, rdd = stats(seg, yrs_train)
            grid_rows.append((direction, T, tr, cagr, mdd, rdd))
            if best is None or rdd > best[0]:
                best = (rdd, direction, T, nav)
    best_rdd, best_dir, best_T, _ = best
    print(f"=== 训练结果：最优阈值 ===")
    print(f"  方向={best_dir}  T={best_T}  训练期 收益/回撤比={best_rdd}")
    # 训练期该阈值明细
    rule = make_rule(best_dir, best_T)
    nav_tr = make_nav(baro_arr, rets, rule)
    seg = nav_tr[train_mask.values if hasattr(train_mask, "values") else train_mask]
    print(f"  训练期表现: 总收益 {stats(seg, yrs_train)[0]}% | "
          f"收益/回撤比 {stats(seg, yrs_train)[3]} | "
          f"空仓占比 {100*float(np.mean([0 if (np.isnan(b) or rule(b)) else 1 for b in baro_arr[train_mask]])):.1f}%")
    # 打印 grid 前几/峰值附近（透明度）
    top = sorted(grid_rows, key=lambda r: -r[5])[:5]
    print("  训练期 Top5 阈值(收益/回撤比):")
    for d, T, tr, cagr, mdd, rdd in top:
        print(f"    {d:10s} T={T:3d}  总收益={tr:6.1f}%  CAGR={cagr:5.1f}%  MDD={mdd:6.1f}%  收益/回撤={rdd}")

    # ---- 样本外回测：用最优阈值 ----
    rule = make_rule(best_dir, best_T)
    nav_full = make_nav(baro_arr, rets, rule)
    spy_nav_full = np.concatenate([[1.0], np.cumprod(1.0 + rets)])

    def slice_seg(nav, mask):
        seg = nav[mask.values if hasattr(mask, "values") else mask]
        return seg / seg[0]

    strat_seg = slice_seg(nav_full, test_mask)
    spy_seg = slice_seg(spy_nav_full, test_mask)
    seg_dates = dates[test_mask]

    print(f"\n=== 样本外回测（{TEST_START.date()}~{TEST_END}, Barometer=真实重建, "
          f"阈值={best_T} 方向={best_dir}）===")
    rows = [
        ("SPY 买入持有 (基准)", *stats(spy_seg, yrs_test)),
        (f"二值SPY(重标定 T={best_T},{best_dir})", *stats(strat_seg, yrs_test)),
    ]
    # 固定阈值对比（VTS 原版逻辑）
    for label, (d, T) in {
        "VTS固定>=85→Cash": ("high_cash", 85),
        "VTS固定<20→Cash": ("low_cash", 20),
    }.items():
        r = make_rule(d, T)
        nv = make_nav(baro_arr, rets, r)
        rows.append((label, *stats(slice_seg(nv, test_mask), yrs_test)))
    df = pd.DataFrame(rows, columns=["策略", "total_return_%", "CAGR_%", "max_drawdown_%", "ret_dd_ratio"])
    df.to_csv(f"{OUT_DIR}/binary_summary_oos.csv", index=False)
    print(df.to_string(index=False))

    # 2020 危机去风险检查
    crisis = (seg_dates >= pd.Timestamp("2020-01-01")) & (seg_dates <= pd.Timestamp("2020-12-31"))
    cash2020 = 0
    for b in baro_arr[test_mask][crisis]:
        cash2020 += 0 if (np.isnan(b) or rule(b)) else 1
    print(f"\n  2020 新冠危机期：重标定策略空仓占比 = {100*cash2020/max(1,crisis.sum()):.0f}%  "
          f"(VTS固定>=85 通常 ~0%，因为 Barometer 极少破 85)")

    # 阈值敏感性（±5 / ±10）
    print("\n=== 阈值敏感性（样本外，固定方向=%s）===" % best_dir)
    sens = []
    for dT in [0, -5, 5, -10, 10]:
        T = max(20, min(95, best_T + dT))
        r = make_rule(best_dir, T)
        nv = make_nav(baro_arr, rets, r)
        sens.append((f"T={T}", *stats(slice_seg(nv, test_mask), yrs_test)))
    sdf = pd.DataFrame(sens, columns=["阈值", "total_return_%", "CAGR_%", "max_drawdown_%", "ret_dd_ratio"])
    print(sdf.to_string(index=False))

    # ---- 出图：样本外 nav ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seg_dates, y=spy_seg * 100, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=seg_dates, y=strat_seg * 100,
                             name=f"二值SPY(重标定 T={best_T})",
                             line=dict(color="#d62728", width=2.5)))
    fig.update_layout(
        title=f"Barometer 二值 SPY 样本外回测 ({TEST_START.date()}~{TEST_END}, "
              f"训练阈值 T={best_T}/{best_dir})",
        xaxis_title="日期", yaxis_title="净值 (起点=100)",
        template="plotly_white", height=640,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    chart_path = f"{OUT_DIR}/binary_chart_oos.html"
    plot_offline(fig, filename=chart_path, auto_open=False)
    print(f"\n图表: {chart_path}")


if __name__ == "__main__":
    run()
