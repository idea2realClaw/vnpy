"""反向工程 VTS Total Portfolio Solution —— Volatility Barometer 复合指标 + 资产轮动回测。

参考 https://www.volatilitytradingstrategies.com/portfolio 及其子策略（第三方评测
greatdaytrading / VTS 官方博客暴露的具体阈值）。VTS 核心是“Volatility Barometer”
（0-100 复合波动率仪表，融合十几项波动率指标），按档位在资产间战术轮动：

  Defensive Rotation（VTS 官方博客原文阈值）:
      Barometer < 20%          -> Cash        （从 Cash 重新进场需 Barometer > 30%，滞后）
      20% - 66%                -> QLD (2x Nasdaq)
      66% - 85%                -> XLU (公用事业)
      > 85%                    -> Cash
  Strategic Tail Risk（第三方评测阈值）:
      0% - 55%                 -> SSO (2x S&P 500)
      55% - 80%                -> IYR (房地产)
      > 80%                    -> VIXM (VIX 中期期货, 做多波动率)

VTS 真实 Barometer 公式保密（十几项指标）。本脚本用两种口径驱动上述轮动：

(A) 真实反向工程 Barometer（主，PRIMARY="real"）：直接复用 vts_barometer_formula.py 的
    build_barometer()，即 16 个等分位指标（VIX 多窗口 + VIX3M + 期限结构 + 已实现/隐含波动
    + VVIX 水平 + VVIX/VIX 比[反向] + VIX9D 水平）的等权滚动百分位，已校验与 VTS 公布锚点
    高度吻合（均值43.3/峰86.3@2020新冠/2017低点13.7）。无未来函数。

(B) 自研透明近似（对比行，仅 VIX/VXV）：
    p1   = VIX 相对近 10 年(2520 日)历史的百分位 rank      (长期水平，危机期逼近100)
    p2   = VIX 相对近 50 日历史的百分位 rank               (短期水平)
    z    = VIX 相对自身 60 日均线的 z-score               (尖峰/陡升)
    term = VIX / VXV  (VXV=^VIX3M, 3 月波动率) 期限结构    (backwardation>1=危机)
    p1_only      = p1*100
    composite    = 0.40*p1+0.15*p2+0.20*z+0.25*term
    crisis_aware = 0.25*p1+0.15*p2+0.30*z+0.30*term

调试发现：composite/crisis_aware 在 2020 新冠仅空仓~13%，危机期没避险（失败）；
p1_only 在 2020 空仓69%/2022 空仓47%+防御46%，危机避险正确。现以真实重建 Barometer 为主、
p1_only/composite/crisis_aware 作对比，验证真实 Barometer 是否更好地复现 VTS 价值主张。

执行约定：信号在 t 日收盘后算出，t+1 日按收盘调仓（close-to-close，与 vix_rank 向量化一致；属乐观口径）。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vts_backtest.py
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

WINDOWS = {
    "2022_2026": (datetime(2022, 1, 3), datetime(2026, 7, 15)),
    "2017_2026": (datetime(2017, 1, 3), datetime(2026, 7, 15)),
}

PRIMARY = "real"   # 主报告模式：真实反向工程 Barometer（含 VVIX+VIX9D，16 分量）


# ----------------------------------------------------------------------------- 数据
def load_close(symbol: str) -> pd.Series:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT datetime, close_price FROM dbbardata "
        f"WHERE symbol='{symbol}' AND exchange='SMART' ORDER BY datetime",
        con,
    )
    con.close()
    df["dt"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    s = df.set_index("dt")["close_price"].astype(float)
    s.index = s.index.normalize()
    return s


def load_all() -> dict:
    out = {}
    for sym in ["SPY", "VIX", "VXV", "QLD", "XLU", "IYR", "VIXM"]:
        out[sym] = load_close(sym)
    return out


# ----------------------------------------------------------------------------- 指标
def pct_rank(series: np.ndarray, window: int) -> np.ndarray:
    """逐点百分位 rank：截至 t（含）窗口内 <= 当前值 的比例。无未来函数。"""
    n = len(series)
    out = np.full(n, np.nan)
    for t in range(1, n):
        w = series[max(0, t - window):t + 1]
        out[t] = float(np.mean(w <= series[t]))
    return out


def build_components(vix: np.ndarray, vxv: np.ndarray) -> dict:
    n = len(vix)
    p1 = pct_rank(vix, 2520)
    p2 = pct_rank(vix, 50)
    ma = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    for t in range(60, n):
        seg = vix[t - 60:t]
        ma[t] = seg.mean()
        sd[t] = seg.std() or 1.0
    z = (vix - ma) / sd
    term = vix / vxv
    p1_s = np.nan_to_num(p1 * 100)
    p2_s = np.nan_to_num(p2 * 100)
    z_s = np.clip(50 + np.nan_to_num(z) * 25, 0, 100)
    term_s = np.clip(50 + (term - 1.0) * 200, 0, 100)
    return {"p1": p1_s, "p2": p2_s, "z": z_s, "term": term_s}


def barometer(mode: str, c: dict) -> np.ndarray:
    if mode == "p1_only":
        return c["p1"]
    if mode == "composite":
        return 0.40 * c["p1"] + 0.15 * c["p2"] + 0.20 * c["z"] + 0.25 * c["term"]
    if mode == "crisis_aware":
        return 0.25 * c["p1"] + 0.15 * c["p2"] + 0.30 * c["z"] + 0.30 * c["term"]
    raise ValueError(mode)


def get_barometer(mode: str, comp: dict, b_real: np.ndarray) -> np.ndarray:
    """统一入口：real 模式返回真实重建 Barometer，其余走自研近似。"""
    if mode == "real":
        return b_real
    return barometer(mode, comp)


# ----------------------------------------------------------------------------- 轮动
def defensive_rotation_alloc(baro: np.ndarray) -> np.ndarray:
    """0=QLD, 1=XLU, 2=Cash。带滞后（<20->Cash, 重进需>30）。"""
    n = len(baro)
    alloc = np.full(n, 2, dtype=int)
    prev = 2
    for t in range(n):
        b = baro[t]
        if b < 20:
            a = 2
        elif b < 30:
            a = prev if prev != 2 else 2
        elif b < 66:
            a = 0
        elif b < 85:
            a = 1
        else:
            a = 2
        alloc[t] = a
        prev = a
    return alloc


def tail_risk_alloc(baro: np.ndarray) -> np.ndarray:
    """0=SSO(2xSPY), 1=IYR, 2=VIXM。"""
    n = len(baro)
    alloc = np.full(n, 0, dtype=int)
    for t in range(n):
        b = baro[t]
        alloc[t] = 0 if b < 55 else (1 if b < 80 else 2)
    return alloc


def portfolio_nav(alloc: np.ndarray, rets: dict, asset_order: list) -> np.ndarray:
    """alloc: 每日资产索引(长度=价格点数)；rets: {asset: 日收益}(长度=价格点数-1)。
    信号在 t-1 日收盘算出 -> 应用于 t-1->t 这段收益(rets[t-1])。无未来函数。"""
    n = len(alloc)
    nav = np.ones(n)
    for t in range(1, n):
        a = alloc[t - 1]
        nav[t] = nav[t - 1] * (1.0 + rets[asset_order[a]][t - 1])
    return nav


# ----------------------------------------------------------------------------- 统计
def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def cagr(nav: np.ndarray, years: float) -> float:
    return float((nav[-1] / nav[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def stats(nav: np.ndarray, years: float) -> tuple:
    return (
        round((nav[-1] / nav[0] - 1.0) * 100, 2),
        round(cagr(nav, years) * 100, 2),
        round(max_drawdown(nav) * 100, 2),
        round((nav[-1] / nav[0] - 1.0) / abs(max_drawdown(nav)), 2),
    )


# ----------------------------------------------------------------------------- 主流程
def run():
    data = load_all()
    common = data["SPY"].index
    for s in ["VIX", "VXV", "QLD", "XLU", "IYR", "VIXM"]:
        common = common.intersection(data[s].index)
    print(f"共同交易日: {len(common)}  ({common[0].date()} ~ {common[-1].date()})")

    spy = data["SPY"].loc[common].to_numpy()
    vix = data["VIX"].loc[common].to_numpy()
    vxv = data["VXV"].loc[common].to_numpy()
    qld = data["QLD"].loc[common].to_numpy()
    xlu = data["XLU"].loc[common].to_numpy()
    iyr = data["IYR"].loc[common].to_numpy()
    vixm = data["VIXM"].loc[common].to_numpy()
    dates = common

    spy_ret = np.diff(spy) / spy[:-1]
    qld_ret = np.diff(qld) / qld[:-1]
    xlu_ret = np.diff(xlu) / xlu[:-1]
    iyr_ret = np.diff(iyr) / iyr[:-1]
    vixm_ret = np.diff(vixm) / vixm[:-1]
    sso_ret = 2.0 * spy_ret

    dr_assets = {"QLD": qld_ret, "XLU": xlu_ret, "Cash": np.zeros_like(qld_ret)}
    tr_assets = {"SSO": sso_ret, "IYR": iyr_ret, "VIXM": vixm_ret}

    comp = build_components(vix, vxv)

    # 真实反向工程 Barometer（含 VVIX+VIX9D，16 分量），对齐到 common 索引；
    # warmup(NaN) 与 VIX9D 缺失日已在中置 0.5 中性，此处再 fillna(50) 保证轮动能跑（=完全持仓）。
    baro_real_series, _, _ = build_barometer()
    b_real = baro_real_series.reindex(common).to_numpy(dtype=float)
    b_real = np.nan_to_num(b_real, nan=50.0)

    # 各模式总览（全样本，看危机避险是否生效）
    print("\n=== 各 Barometer 模式：危机期空仓占比（验证避险）===")
    yrs = pd.Series(dates).dt.year.values
    for mode in ["real", "p1_only", "composite", "crisis_aware"]:
        b = get_barometer(mode, comp, b_real)
        a = defensive_rotation_alloc(b)
        line = f"  {mode:12s} Cash总占比={100*(a==2).mean():.1f}%"
        for y in [2018, 2020, 2022]:
            m = yrs == y
            line += f" | {y} Cash={100*(a[m]==2).mean():.0f}%"
        print(line)

    # 主报告：双窗口， PRIMARY 模式
    results = {}
    for wlabel, (sd, ed) in WINDOWS.items():
        mask = (dates >= pd.Timestamp(sd)) & (dates <= pd.Timestamp(ed))
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        i0, i1 = idx[0], idx[-1]
        years = (dates[i1] - dates[i0]).days / 365.25

        b_full = get_barometer(PRIMARY, comp, b_real)
        dr_alloc = defensive_rotation_alloc(b_full)
        tr_alloc = tail_risk_alloc(b_full)
        dr_nav_full = portfolio_nav(dr_alloc, dr_assets, ["QLD", "XLU", "Cash"])
        tr_nav_full = portfolio_nav(tr_alloc, tr_assets, ["SSO", "IYR", "VIXM"])
        total_nav_full = np.sqrt(dr_nav_full * tr_nav_full)

        spy_nav_full = np.concatenate([[1.0], np.cumprod(1.0 + spy_ret)])
        qld_nav_full = np.concatenate([[1.0], np.cumprod(1.0 + qld_ret)])
        sso_nav_full = np.concatenate([[1.0], np.cumprod(1.0 + sso_ret)])

        def slice_nav(nav):
            seg = nav[i0:i1 + 1]
            return seg / seg[0]

        spy_seg = slice_nav(spy_nav_full)
        qld_seg = slice_nav(qld_nav_full)
        sso_seg = slice_nav(sso_nav_full)
        dr_seg = slice_nav(dr_nav_full)
        tr_seg = slice_nav(tr_nav_full)
        total_seg = slice_nav(total_nav_full)
        seg_dates = dates[i0:i1 + 1]

        rows = [
            ("SPY 买入持有 (基准)", *stats(spy_seg, years)),
            ("QLD 买入持有 (2x Nasdaq)", *stats(qld_seg, years)),
            ("SSO 买入持有 (2x S&P, 合成)", *stats(sso_seg, years)),
            ("VTS Defensive Rotation (反向工程)", *stats(dr_seg, years)),
            ("VTS Strategic Tail Risk (反向工程)", *stats(tr_seg, years)),
            ("VTS Total Portfolio (两策略等权)", *stats(total_seg, years)),
        ]
        df = pd.DataFrame(rows, columns=["策略", "total_return_%", "CAGR_%", "max_drawdown_%", "ret_dd_ratio"])
        df.to_csv(f"{OUT_DIR}/vts_summary_{wlabel}.csv", index=False)

        results[wlabel] = dict(dates=seg_dates, spy=spy_seg, qld=qld_seg, sso=sso_seg,
                               dr=dr_seg, tr=tr_seg, total=total_seg, df=df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seg_dates, y=spy_seg * 100, name="SPY 买入持有",
                                 line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=seg_dates, y=qld_seg * 100, name="QLD 买入持有(2x Nasdaq)",
                                 line=dict(color="#ff7f0e", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=seg_dates, y=dr_seg * 100, name="VTS Defensive Rotation",
                                 line=dict(color="#2ca02c", width=2)))
        fig.add_trace(go.Scatter(x=seg_dates, y=tr_seg * 100, name="VTS Strategic Tail Risk",
                                 line=dict(color="#9467bd", width=2)))
        fig.add_trace(go.Scatter(x=seg_dates, y=total_seg * 100, name="VTS Total Portfolio(等权)",
                                 line=dict(color="#d62728", width=2.5)))
        fig.update_layout(
            title=f"反向工程 VTS 组合策略净值对比 ({wlabel.replace('_', '-')}, 起点=100, Barometer=真实重建)",
            xaxis_title="日期", yaxis_title="净值 (起点=100)",
            template="plotly_white", height=640,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        chart_path = f"{OUT_DIR}/vts_chart_{wlabel}.html"
        plot_offline(fig, filename=chart_path, auto_open=False)

        print(f"\n=== 窗口 {wlabel} ({sd.date()} ~ {ed.date()}, {years:.1f}年) | Barometer={PRIMARY} ===")
        print(df.to_string(index=False))
        print(f"图表: {chart_path}")

    # 多模式对比（仅 2017-2026 窗口，主表）
    print("\n=== 多 Barometer 模式对比（2017-2026 窗口，Total Portfolio 等权）===")
    cmp_rows = []
    idx = np.where((dates >= pd.Timestamp(WINDOWS["2017_2026"][0])) &
                   (dates <= pd.Timestamp(WINDOWS["2017_2026"][1])))[0]
    i0, i1 = idx[0], idx[-1]
    years = (dates[i1] - dates[i0]).days / 365.25
    spy_nav_full = np.concatenate([[1.0], np.cumprod(1.0 + spy_ret)])
    for mode in ["real", "p1_only", "composite", "crisis_aware"]:
        b = get_barometer(mode, comp, b_real)
        dr_nav = portfolio_nav(defensive_rotation_alloc(b), dr_assets, ["QLD", "XLU", "Cash"])
        tr_nav = portfolio_nav(tail_risk_alloc(b), tr_assets, ["SSO", "IYR", "VIXM"])
        total = np.sqrt(dr_nav * tr_nav)
        seg = total[i0:i1 + 1] / total[i0]
        cmp_rows.append((mode, *stats(seg, years)))
    cmp_df = pd.DataFrame(cmp_rows, columns=["Barometer模式", "total_return_%", "CAGR_%", "max_drawdown_%", "ret_dd_ratio"])
    cmp_df.to_csv(f"{OUT_DIR}/vts_mode_compare.csv", index=False)
    print(cmp_df.to_string(index=False))

    return results


if __name__ == "__main__":
    run()
