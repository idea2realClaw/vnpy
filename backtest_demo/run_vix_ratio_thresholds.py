"""
run_vix_ratio_thresholds.py —— 扫描阈值 thr ∈ {1.6,1.7,1.8,1.9,2.0} 的收益/回撤/夏普

方向固定 high_cash（ratio>thr 空仓），合成 ratio=VIX_ma3/VIX_ma30。
报告两段窗口：
  全样本 2011-2025
  样本外 2016-2026
指标：总收益(%)、最大回撤(%)、年化夏普(无风险=0)、空仓占比(%)。
并附 SPY 买入持有基准。

用法（仓库根目录执行）：
    PYTHONPATH=/Users/zhuxiaodong/Documents/GitRepo/vnpy \
      /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ratio_thresholds.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"
THRS = [1.6, 1.7, 1.8, 1.9, 2.0]
DIR = "high_cash"
WINS = {"全样本 2011-2025": ("2011-01-01", "2025-12-31"),
        "样本外 2016-2026": ("2016-01-01", "2026-07-15")}


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


def metrics(thr, ratio, ret, valid, dates, d0, d1):
    nav, pos = nav_for(thr, DIR, ratio, ret, valid)
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    seg = nav[ii[0]:ii[-1] + 1]
    tot = seg[-1] / seg[0] - 1.0
    peak = np.maximum.accumulate(seg)
    mdd = (seg / peak - 1.0).min()
    # 窗口内日收益（t 对应 ret[t]）
    tidx = np.arange(ii[0], ii[-1])
    day_ret = pos[tidx] * ret[tidx]
    sharpe = day_ret.mean() / day_ret.std(ddof=1) * np.sqrt(252) if day_ret.std(ddof=1) > 0 else 0.0
    cash = float(np.mean(pos[tidx] == 0.0))
    return tot * 100, mdd * 100, sharpe, cash * 100


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

    # SPY 基准
    spy_metrics = {}
    for wname, (d0, d1) in WINS.items():
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        tidx = np.arange(ii[0], ii[-1])
        dr = ret[tidx]
        tot = (s[ii[-1]] / s[ii[0]] - 1.0) * 100
        peak = np.maximum.accumulate(s[ii[0]:ii[-1] + 1])
        mdd = ((s[ii[0]:ii[-1] + 1] / peak) - 1.0).min() * 100
        shp = dr.mean() / dr.std(ddof=1) * np.sqrt(252)
        spy_metrics[wname] = (tot, mdd, shp, 0.0)

    rows = []
    for thr in THRS:
        row = {"阈值": f"{thr:.1f}"}
        for wname, (d0, d1) in WINS.items():
            tot, mdd, shp, cash = metrics(thr, ratio, ret, valid, dates, d0, d1)
            row[f"{wname} 收益%"] = round(tot, 1)
            row[f"{wname} 回撤%"] = round(mdd, 1)
            row[f"{wname} 夏普"] = round(shp, 2)
            row[f"{wname} 空仓%"] = round(cash, 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    for wname in WINS:
        tot, mdd, shp, _ = spy_metrics[wname]
        print(f"SPY 基准 [{wname}]：收益 {tot:.1f}%  回撤 {mdd:.1f}%  夏普 {shp:.2f}")

    # CSV
    df.to_csv(f"{OUT_DIR}/vix_ratio_thresholds.csv", index=False)

    # HTML 表格
    header = ["阈值"] + [c for c in df.columns if c != "阈值"]
    cells = [[str(x) for x in df["阈值"]]]
    for c in header[1:]:
        cells.append([str(x) for x in df[c]])
    # 追加 SPY 基准行
    spy_row = ["SPY"]
    for wname in WINS:
        tot, mdd, shp, _ = spy_metrics[wname]
        spy_row += [f"{tot:.1f}", f"{mdd:.1f}", f"{shp:.2f}", "0.0"]
    cells[0].append("SPY")
    for j, c in enumerate(header[1:]):
        cells[j + 1].append(spy_row[j + 1])

    fig = go.Figure(data=[go.Table(
        header=dict(values=header, fill_color="#2c3e50", font=dict(color="white", size=12),
                    align="center"),
        cells=dict(values=cells, fill_color=[["#ffffff", "#f2f2f2"] * (len(cells[0]) // 2 + 1)],
                   align="center", font=dict(size=11)),
    )])
    fig.update_layout(
        title="VIX_ma3/VIX_ma30 比值二值策略 · 阈值扫描（方向=high_cash, 无风险=0）",
        height=360, template="plotly_white",
    )
    path = f"{OUT_DIR}/vix_ratio_thresholds.html"
    plot_offline(fig, filename=path, auto_open=False)
    print(f"\n交付：{path}\n      {OUT_DIR}/vix_ratio_thresholds.csv")


if __name__ == "__main__":
    main()
