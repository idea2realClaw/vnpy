"""
vts_barometer13_sim.py —— 重建并模拟 VTS Volatility Barometer（13 因子）

重建依据（VTS 官方文章 + 用户提供的信息，均逐条可考据）：
  · 13 个波动率指标等权合成，输出 0%–100% 读数（官方 inception=2011-01-01，因需 5 年数据排百分位）。
  · 每个指标单独做 5 年滚动百分位（T=1260），等权平均 → 单一读数。
  · 数学特性：须 13 指标同日达极值才=0/100；Traders VRP、Cash VIX Oscillator 在极低波动时呈反向，
    故现实中几乎不归零。长期均值 ≈ 46.62%（2011 至今）。
  · 披露锚点：历史极低 2017-07-21 = 13.82%；极高 2020-03-12 = 90.95%（旧算法上限未破 91%）。
  · 三套子策略变体（错开信号权重）：
      Strategic Tail Risk  → 完整 Barometer（等权）
      Defensive Rotation   → 加重 VIX 期货/期限结构类（M1:M2、roll yield）+ VRP
      Tactical Balanced    → 加重 Cash VIX 期限结构类（Cash VIX Oscillator）

数据来源（SQLite ~/.vntrader/database.db）：
  VIX / VIX9D / VXV(VIX3M) / VVIX / VIXM(中期期货ETF, 代理 VIX6M) / SPY(算 HV 已实现波动)
  无逐月 VIX 期货 M1/M2、VX30 的盘中数据 → 用 VIX/VXV/VIXM 派生代理，已在因子表注明。

用法（仓库根目录执行）：
    PYTHONPATH=/Users/zhuxiaodong/Documents/GitRepo/vnpy \
      /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/vts_barometer13_sim.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline

from backtest_demo.run_vts_backtest import load_close

OUT = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"
W = 1260  # 5 年滚动百分位窗口


def pct_rank(s, w=W):
    s = pd.Series(s, dtype=float)
    # 当前值在 5 年窗口内的百分位：窗口内有多少值 <= 当前值（高波动→高分）
    return s.rolling(w, min_periods=w).apply(lambda x: float((x[-1] >= x).mean()), raw=True)


def main():
    # ---------- 数据 ----------
    vix = load_close("VIX")
    vix9d = load_close("VIX9D")
    vxv = load_close("VXV")
    vvix = load_close("VVIX")
    vixm = load_close("VIXM")
    spy = load_close("SPY")
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

    # ---------- 13 因子原始序列 ----------
    f = {}
    f["VIX_spot"] = VIX                                   # 30日隐含波动
    f["VIX9D"] = VIX9D                                    # 9日短端隐含波动
    f["VIX3M_VXV"] = VXV                                  # 3个月隐含波动
    f["VIX6M_VIXM"] = VIXM                                # 中期期货ETF(代理 VIX6M/VXMT)
    f["VVIX"] = VVIX                                      # 波动率之波动率
    f["VIX_VIX3M_ratio"] = VIX / VXV                      # 中短期期限结构交叉(>1=应激)
    f["VIX9D_VIX_ratio"] = VIX9D / VIX                    # 短端 vs 30日(>1=应激)
    f["M1_M2_contango"] = VIX - VXV                       # 近月期限升水(代理 M1:M2, 倒挂=应激)
    f["VX30_VIX_rollyield"] = VIX - VIXM                  # 现货 vs 中期期货(现货>中期=backwardation=应激)
    f["Simple_VRP"] = VIX - HV20                          # 波动率风险溢价 VIX-HV20(扩大=应激)
    # Traders VRP = WMA(VX30 - HV5)，VX30 以 VXV 代理
    w = np.array([1, 2, 3, 4, 5], float)
    w /= w.sum()
    f["Traders_VRP"] = (VXV - HV5).rolling(5).apply(lambda x: float(np.sum(w * x)), raw=True)
    f["Cash_VIX_Oscillator"] = (VIX - sma50) / sma50      # VIX 相对其均线(极低波动时反向, 不归零)
    f["VTS_Cash_VIX_Oscillator"] = VIX - VXV              # 期限结构版 Cash VIX(应激时为正)

    group = {
        "VIX_spot": "level", "VIX9D": "level", "VIX3M_VXV": "level",
        "VIX6M_VIXM": "level", "VVIX": "level",
        "VIX_VIX3M_ratio": "term", "VIX9D_VIX_ratio": "term",
        "M1_M2_contango": "term", "VX30_VIX_rollyield": "term",
        "Simple_VRP": "vrp", "Traders_VRP": "vrp",
        "Cash_VIX_Oscillator": "cash", "VTS_Cash_VIX_Oscillator": "cash",
    }
    src = {
        "VIX_spot": "VIX (真实)", "VIX9D": "VIX9D (真实)",
        "VIX3M_VXV": "VXV (真实)", "VIX6M_VIXM": "VIXM 中期期货ETF (代理 VXMT)",
        "VVIX": "VVIX (真实)", "VIX_VIX3M_ratio": "VIX/VXV (派生)",
        "VIX9D_VIX_ratio": "VIX9D/VIX (派生)", "M1_M2_contango": "VIX-VXV (代理 M1:M2)",
        "VX30_VIX_rollyield": "VIX-VIXM (代理 VX30:VIX)", "Simple_VRP": "VIX-HV20(SPY) (派生)",
        "Traders_VRP": "WMA(VXV-HV5) (代理 VX30-HV5)", "Cash_VIX_Oscillator": "z(VIX vs SMA50) (派生)",
        "VTS_Cash_VIX_Oscillator": "VIX-VXV 期限结构 (派生)",
    }

    # ---------- 百分位 + 合成 ----------
    pct = {k: pct_rank(v) for k, v in f.items()}
    P = pd.DataFrame(pct)
    baro_eq = P.mean(axis=1, skipna=True) * 100.0          # 等权（Strategic Tail Risk）

    # 子策略变体权重
    def weighted(grp_boost):
        wsum = np.zeros(len(P))
        acc = np.zeros(len(P))
        for k in P.columns:
            wk = 2.0 if group[k] in grp_boost else 1.0
            mask = P[k].notna().to_numpy()
            acc += np.where(mask, P[k].to_numpy() * wk, 0.0)
            wsum += np.where(mask, wk, 0.0)
        safe = np.divide(acc, wsum, out=np.full_like(acc, np.nan), where=wsum > 0)
        return pd.Series(safe * 100.0, index=P.index)

    baro_def = weighted({"term", "vrp"})       # Defensive Rotation
    baro_bal = weighted({"cash"})              # Tactical Balanced

    # 新算法（2025-05 可视化微调）：min-max 重标定到 0-100，使 COVID 显示=100
    lo, hi = baro_eq.min(), baro_eq.max()
    baro_new = (baro_eq - lo) / (hi - lo) * 100.0

    # ---------- 校验锚点 ----------
    def val_on(s, d):
        sub = s[s.index >= d]
        return float(sub.iloc[0]) if len(sub) else np.nan

    d_low, d_high = "2017-07-21", "2020-03-12"
    sim_low = val_on(baro_eq, d_low)
    sim_high = val_on(baro_eq, d_high)
    sim_mean = float(baro_eq[baro_eq.index >= "2011-01-01"].mean())
    print("=== 校验：模拟值 vs VTS 披露锚点 ===")
    print(f"  极低 2017-07-21 : 模拟 {sim_low:.2f}%  | 披露 13.82%  | 偏差 {sim_low-13.82:+.2f}")
    print(f"  极高 2020-03-12 : 模拟 {sim_high:.2f}%  | 披露 90.95%  | 偏差 {sim_high-90.95:+.2f}")
    print(f"  长期均值(2011~) : 模拟 {sim_mean:.2f}%  | 披露 46.62%  | 偏差 {sim_mean-46.62:+.2f}")

    # ---------- 子策略当前读数 ----------
    print("\n=== 三套子策略变体最新读数 ===")
    last = baro_eq.index[-1]
    print(f"  {last.date()}  Strategic Tail Risk(等权)={baro_eq.iloc[-1]:.1f}  "
          f"Defensive Rotation={baro_def.iloc[-1]:.1f}  Tactical Balanced={baro_bal.iloc[-1]:.1f}")

    # ---------- CSV：因子定义 + 当前百分位 ----------
    rows = []
    for k in f:
        cur = P[k].iloc[-1]
        rows.append({"因子": k, "类别": group[k], "数据来源": src[k],
                     "当前百分位%": round(float(cur) * 100, 1) if pd.notna(cur) else None})
    pd.DataFrame(rows).to_csv(f"{OUT}/vts_barometer13_factors.csv", index=False)

    # ---------- HTML ----------
    dts = baro_eq.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dts, y=baro_eq, name="Barometer 等权(13因子, 旧算法)",
                             line=dict(color="#1f2d5a", width=2.2)))
    fig.add_trace(go.Scatter(x=dts, y=baro_new, name="新算法(2025-05重标定, COVID→100)",
                             line=dict(color="#7b1fa2", width=1.3, dash="dot")))
    fig.add_trace(go.Scatter(x=dts, y=baro_def, name="Defensive Rotation(重期货/VRP)",
                             line=dict(color="#d62728", width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=dts, y=baro_bal, name="Tactical Balanced(重Cash VIX)",
                             line=dict(color="#2ca02c", width=1.4, dash="dash")))
    fig.add_hrect(y0=0, y1=33.33, fillcolor="green", opacity=0.10,
                  line_width=0, annotation_text="绿区(最低1/3)", annotation_position="top left")
    fig.add_hrect(y0=66.67, y1=100, fillcolor="red", opacity=0.10,
                  line_width=0, annotation_text="红区(最高1/3)", annotation_position="top left")
    # 锚点标注
    fig.add_trace(go.Scatter(x=[pd.Timestamp(d_low)], y=[sim_low], mode="markers+text",
                             name="披露极低 13.82% (2017-07-21)",
                             marker=dict(color="green", size=9, symbol="circle"),
                             text=[f"模拟 {sim_low:.1f}% / 披露 13.82%"],
                             textposition="bottom right"))
    fig.add_trace(go.Scatter(x=[pd.Timestamp(d_high)], y=[sim_high], mode="markers+text",
                             name="披露极高 90.95% (2020-03-12)",
                             marker=dict(color="red", size=9, symbol="circle"),
                             text=[f"模拟 {sim_high:.1f}% / 披露 90.95%"],
                             textposition="top right"))
    fig.update_layout(title="VTS Volatility Barometer（13因子重建模拟, 2011-2026, 绿/红区=最低/最高1/3）"
                             f"<br>长期均值 模拟 {sim_mean:.1f}% vs 披露 46.62%",
                      xaxis_title="日期", yaxis_title="Barometer 读数 (0-100)",
                      template="plotly_white", height=640, yaxis_range=[0, 100],
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    path = f"{OUT}/vts_barometer13_sim.html"
    plot_offline(fig, filename=path, auto_open=False)
    print(f"\n交付：{path}\n      {OUT}/vts_barometer13_factors.csv")


if __name__ == "__main__":
    main()
