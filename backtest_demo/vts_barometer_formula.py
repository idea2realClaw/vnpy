"""
VTS Volatility Barometer —— 反向工程公式实现 + 与 VTS 公布锚点校验
=================================================================
公式（见 vts_barometer_formula.md）：
  Step1  r_i(t) = (1/T) * Σ 1{ m_i(t-k+1) < m_i(t) }      T=1260（≈5年滚动）
  Step2  q_i(t) = r_i(t) 或 1-r_i(t)（反向指标）
  Step3  Barometer(t) = 100 * Σ w_i q_i(t) / Σ w_i         （等权）

数据：SQLite（~/.vntrader/database.db）中 VIX / VXV(=VIX3M) / SPY
缺失的 ^VVIX / ^VIX9D 用可计算代理补足 13 个指标，注释标明。

输出：
  - 打印与 VTS 公布锚点的对比表（均值/极值/尾部占比/Volpocalypse 前后）
  - 生成 vts_barometer_chart.html（重建 Barometer 时序 + 锚点标注）
"""
import sqlite3
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DB_PATH = "/Users/zhuxiaodong/.vntrader/database.db"
T = 1260  # ≈5 年滚动窗口


def load(symbol: str) -> pd.Series:
    con = sqlite3.connect(DB_PATH)
    d = pd.read_sql(
        f"SELECT datetime, close_price FROM dbbardata WHERE symbol='{symbol}' AND exchange='SMART' ORDER BY datetime",
        con,
    )
    con.close()
    d["dt"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None)
    d["dt"] = d["dt"].dt.normalize()
    s = d.set_index("dt")["close_price"].astype(float)
    return s


def rolling_pct(x: np.ndarray) -> np.ndarray:
    """r_i(t) = 滚动窗口内 严格小于 当前值 的比例（含当日）。"""
    n = len(x)
    out = np.full(n, np.nan)
    if n < T:
        return out
    sw = sliding_window_view(x, T)            # (n-T+1, T)
    cur = sw[:, -1:]                           # 当前值广播
    ranks = (sw < cur).mean(axis=1)            # 窗口内百分位
    out[T - 1:] = ranks
    return out


def realized_vol(rets: np.ndarray, win: int) -> np.ndarray:
    n = len(rets)
    out = np.full(n, np.nan)
    for t in range(win, n):
        out[t] = np.std(rets[t - win:t]) * np.sqrt(252)
    return out


def sma(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    sw = sliding_window_view(x, win)
    out[win - 1:] = sw.mean(axis=1)
    return out


def build_barometer():
    vix = load("VIX")
    vxv = load("VXV")          # 即 VIX3M
    spy = load("SPY")
    idx = vix.index.intersection(vxv.index).intersection(spy.index)
    vix, vxv, spy = vix.loc[idx], vxv.loc[idx], spy.loc[idx]
    v = vix.to_numpy(); x = vxv.to_numpy(); s = spy.to_numpy()
    n = len(v)
    dates = np.array(idx)

    spy_ret = np.zeros(n); spy_ret[1:] = np.diff(np.log(s))
    rv21 = realized_vol(spy_ret, 21)                       # 已实现波动率
    rv60 = realized_vol(spy_ret, 60)
    vix_sma20 = sma(v, 20)
    vix_sma60 = sma(v, 60)

    # ---- 13 个指标（各自转滚动百分位；标 [C] 为反向指标）----
    metrics = {}
    metrics["VIX_level"]   = rolling_pct(v)
    metrics["VIX_252d"]    = rolling_pct(v)                # 长窗口已由 T 覆盖，此处作不同尺度占位
    metrics["VIX63d_pct"]  = rolling_pct(v)                # 占位（实际应短窗，见下注）
    # 用不同滑动窗口覆盖"不同尺度"：重算短窗百分位
    def rp_win(x, w):
        nn = len(x); out = np.full(nn, np.nan)
        if nn < w: return out
        sw = sliding_window_view(x, w); cur = sw[:, -1:]
        out[w - 1:] = (sw < cur).mean(axis=1); return out
    metrics["VIX_level"]   = rp_win(v, T)
    metrics["VIX_63d"]     = rp_win(v, 63)
    metrics["VIX_21d"]     = rp_win(v, 21)
    metrics["VIX3M_level"] = rp_win(x, T)
    metrics["VIX_minus_VIX3M"] = rp_win(v - x, T)          # 期限结构斜率
    metrics["term_ratio"]  = rp_win(v / x, T)              # VIX/VIX3M 升贴水
    metrics["VIX_z60"]     = rp_win((v - vix_sma60) / (vix_sma60.std() or 1), T)
    metrics["VIX_osc"]     = rp_win(v / vix_sma20 - 1.0, T)   # [C] VIX Oscillator 反向
    metrics["VIX_chg5"]    = rp_win(np.diff(np.log(v), prepend=np.nan) * 100, T)  # 5日%变
    metrics["RV21"]        = rp_win(rv21, T)
    metrics["RV60"]        = rp_win(rv60, T)
    metrics["VRP"]         = rp_win(v - rv21, T)              # 隐含-已实现
    metrics["VRP_osc"]     = rp_win((v - rv21) / (rv21 + 1e-9), T)  # [C] VRP 反向代理

    contrarian = {"VIX_osc", "VRP_osc"}
    q = []
    for name, r in metrics.items():
        rr = np.nan_to_num(r, nan=0.5)
        q.append(1.0 - rr if name in contrarian else rr)
    q = np.array(q)
    baro = 100.0 * q.mean(axis=0)
    # 屏蔽 warmup：长期百分位窗口(T)未满前所有指标不可信，置 NaN 避免污染统计
    if n >= T:
        baro[:T - 1] = np.nan

    return pd.Series(baro, index=idx), dates, baro


def anchor_compare(baro: pd.Series):
    vals = baro.dropna()
    d = vals.index
    v = vals.to_numpy()
    pre = (d >= pd.Timestamp("2011-02-05")) & (d <= pd.Timestamp("2018-02-05"))
    post = (d > pd.Timestamp("2018-02-05")) & (d <= pd.Timestamp("2025-02-05"))
    print("=== 重建 Barometer vs VTS 公布锚点 ===")
    print(f"{'校验项':<22}{'VTS公布':>12}{'重建值':>12}")
    print(f"{'全样本均值':<22}{'46.62%':>12}{np.mean(v):>11.2f}%")
    print(f"{'历史最低':<22}{'13.82%':>12}{np.min(v):>11.2f}%  @ {d[np.argmin(v)].date()}")
    print(f"{'历史最高':<22}{'90.95%':>12}{np.max(v):>11.2f}%  @ {d[np.argmax(v)].date()}")
    print(f"{'<20% 占比':<22}{'5.2%':>12}{100*np.mean(v<20):>11.1f}%")
    print(f"{'>80% 占比':<22}{'5.6%':>12}{100*np.mean(v>80):>11.1f}%")
    print(f"{'Volpocalypse前7年均值':<22}{'39.94%':>12}{np.mean(v[pre]):>11.2f}%")
    print(f"{'Volpocalypse后7年均值':<22}{'53.31%':>12}{np.mean(v[post]):>11.2f}%")


def main():
    baro, dates, _ = build_barometer()
    anchor_compare(baro)

    # 出图：重建 Barometer 时序 + 锚点
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baro.index, y=baro.values, name="重建 Barometer",
                             line=dict(color="#1f77b4", width=1.5)))
    for lvl, lab, col in [(46.62, "长期均值 46.62%", "#888"),
                          (20, "20% 线", "#2ca02c"),
                          (80, "80% 线", "#d62728")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=col, annotation_text=lab)
    # 锚点标注
    fig.add_annotation(x="2017-07-21", y=13.82, text="VTS最低 13.82%", showarrow=True, arrowhead=1)
    fig.add_annotation(x="2020-03-12", y=90.95, text="VTS最高 90.95%", showarrow=True, arrowhead=1)
    fig.update_layout(title="反向工程 VTS Volatility Barometer（蓝=重建；点线=VTS公布锚点）",
                      xaxis_title="日期", yaxis_title="Barometer (0-100)",
                      template="plotly_white", height=600)
    out = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo/vts_barometer_chart.html"
    fig.write_html(out)
    print(f"\n图表已保存: {out}")


if __name__ == "__main__":
    main()
