"""
run_vix_ratio_final.py —— 精简交付版

只保留两条净值曲线：
    1) VIX_ma3/VIX_ma30 二值择时（high_cash, thr=2.0）
    2) SPY 买入持有基准

页面最上方列出「最近五个交易日」的：
    VIX、VIX_MA3、VIX_MA30、VIX_MA3/VIX_MA30，
    以及依据 ratio 与 thr=2.0 判定的当日仓位（空仓 / 100%满仓）。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ratio_final.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline
from plotly.subplots import make_subplots

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"
THR = 2.0


def main():
    # ---- 数据 ----
    vix = load_close("VIX")
    spy = load_close("SPY")
    idx = vix.index.intersection(spy.index)
    VIX = vix.loc[idx].astype(float)
    SPY = spy.loc[idx].astype(float)
    dates = idx.to_numpy()
    s = SPY.to_numpy(float)
    ret = np.diff(s) / s[:-1]
    v = VIX.to_numpy(float)

    # ---- 平滑 ----
    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().to_numpy()
    ratio = ma3 / ma30  # 合成指标：短期/长期

    # ---- 二值择时净值（thr=2.0, high_cash）----
    n = len(ret)
    valid = ~np.isnan(ratio[:n]) & (dates[:n] >= np.datetime64("2011-01-01"))
    pos = np.ones(n)
    for t in range(n):
        if not valid[t] or np.isnan(ratio[t]):
            pos[t] = 1.0
        else:
            pos[t] = 0.0 if ratio[t] > THR else 1.0
    nav = [100.0]
    spy_nav = [100.0]
    for t in range(n):
        nav.append(nav[-1] * (1.0 + pos[t] * ret[t]))
        spy_nav.append(spy_nav[-1] * (1.0 + ret[t]))
    nav = np.array(nav)      # 长度 n+1，nav[i] 对应 dates[i]
    spy_nav = np.array(spy_nav)

    # ---- 顶部：最近五日 VIX 指标 + 仓位信号 ----
    # ratio 与 ma3/ma30 与 dates 等长（前 29 个为 NaN）
    last5 = []
    # 取最近 5 个 ratio 非空的交易日
    good = [i for i in range(len(dates)) if not np.isnan(ratio[i])]
    for j in good[-5:]:
        r = ratio[j]
        sig = "空仓" if (not np.isnan(r) and r > THR) else "100%满仓"
        last5.append({
            "date": pd.Timestamp(dates[j]).strftime("%Y-%m-%d"),
            "vix": v[j],
            "ma3": ma3[j],
            "ma30": ma30[j],
            "ratio": r,
            "signal": sig,
        })

    # ---- 净值分段（起点=100）----
    def seg(nav_arr, d0, d1):
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        seg = nav_arr[ii[0]:ii[-1] + 1]
        return seg / seg[0], dates[ii[0]:ii[-1] + 1]

    (full_nav, full_d), (spy_full, _) = seg(nav, "2011-01-01", "2026-07-20"), seg(spy_nav, "2011-01-01", "2026-07-20")
    (oos_nav, oos_d), (spy_oos, _) = seg(nav, "2016-01-01", "2026-07-20"), seg(spy_nav, "2016-01-01", "2026-07-20")

    # ---- 顶部 HTML 表 ----
    def fmt(x, d=2):
        return f"{x:.{d}f}"
    rows = "".join(
        f"<tr><td>{r['date']}</td><td>{fmt(r['vix'])}</td><td>{fmt(r['ma3'])}</td>"
        f"<td>{fmt(r['ma30'])}</td><td>{fmt(r['ratio'])}</td>"
        f"<td class='{'cash' if r['signal']=='空仓' else 'full'}'>{r['signal']}</td></tr>"
        for r in last5
    )
    table_html = f"""
    <div class='panel'>
      <h3>最近五个交易日 VIX 指标与仓位信号（阈值 thr={THR:.1f}）</h3>
      <table>
        <thead><tr><th>日期</th><th>VIX</th><th>VIX_MA3</th><th>VIX_MA30</th>
        <th>VIX_MA3 / VIX_MA30</th><th>信号</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class='note'>信号判定：ratio &gt; {THR:.1f} → 空仓；否则 100% 满仓 SPY。</p>
    </div>
    """

    # ---- 净值图（两面板：全样本 + 样本外）----
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        subplot_titles=("全样本 2011-2026（起点=100）", "样本外 2016-2026（起点=100）"),
        vertical_spacing=0.12, row_heights=[0.5, 0.5],
    )
    fig.add_trace(go.Scatter(x=full_d, y=full_nav, name="VIX比值 thr=2.0",
                             line=dict(color="#d62728", width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=full_d, y=spy_full, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=oos_d, y=oos_nav, name="VIX比值 thr=2.0",
                             line=dict(color="#d62728", width=2.4), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=oos_d, y=spy_oos, name="SPY 买入持有",
                             line=dict(color="#1f77b4", width=2), showlegend=False), row=2, col=1)
    fig.update_layout(
        title="VIX_ma3/VIX_ma30 二值择时（thr=2.0） vs SPY 买入持有",
        template="plotly_white", height=860,
        legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="left", x=0),
        margin=dict(t=70, b=70),
    )
    fig.update_yaxes(title_text="净值（起点=100）", row=1, col=1)
    fig.update_yaxes(title_text="净值（起点=100）", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)

    plot_div = plot_offline(fig, include_plotlyjs="cdn", output_type="div", auto_open=False)

    # 关键绩效
    def metrics(nav_arr, d0, d1):
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        seg = nav_arr[ii[0]:ii[-1] + 1]
        tot = seg[-1] / seg[0] - 1.0
        peak = np.maximum.accumulate(seg)
        mdd = (seg / peak - 1.0).min()
        return tot, mdd
    mt, mdt = metrics(nav, "2011-01-01", "2026-07-20")
    mo, mdo = metrics(nav, "2016-01-01", "2026-07-20")
    st, sdt = metrics(spy_nav, "2011-01-01", "2026-07-20")
    so, sdo = metrics(spy_nav, "2016-01-01", "2026-07-20")

    perf_html = f"""
    <div class='panel'>
      <h3>关键绩效</h3>
      <table>
        <thead><tr><th>窗口</th><th>VIX比值 thr=2.0 收益</th><th>最大回撤</th>
        <th>SPY 收益</th><th>SPY 最大回撤</th></tr></thead>
        <tbody>
          <tr><td>全样本 2011-2026</td><td>{mt*100:.1f}%</td><td>{mdt*100:.1f}%</td>
              <td>{st*100:.1f}%</td><td>{sdt*100:.1f}%</td></tr>
          <tr><td>样本外 2016-2026</td><td>{mo*100:.1f}%</td><td>{mdo*100:.1f}%</td>
              <td>{so*100:.1f}%</td><td>{sdo*100:.1f}%</td></tr>
        </tbody>
      </table>
      <p class='note'>信号 t-1 日收盘算出 → 决定 t 日仓位（无日内前视）。</p>
    </div>
    """

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>VIX比值二值择时 thr=2.0 vs SPY</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; color:#222; }}
  h2 {{ color:#1f2d5a; }}
  .panel {{ background:#fafbfc; border:1px solid #e5e8ee; border-radius:10px; padding:14px 18px; margin:14px 0; }}
  table {{ border-collapse: collapse; width:100%; font-size:14px; }}
  th, td {{ border:1px solid #e1e5ec; padding:8px 10px; text-align:center; }}
  th {{ background:#eef2f8; color:#33415c; }}
  tbody tr:nth-child(even) {{ background:#f6f8fb; }}
  td.cash {{ color:#c0392b; font-weight:700; }}
  td.full {{ color:#1e8449; font-weight:700; }}
  .note {{ color:#6b7280; font-size:12px; margin:8px 0 0; }}
</style></head>
<body>
  <h2>VIX 比值二值择时策略（VIX_MA3 / VIX_MA30，thr=2.0） vs SPY 买入持有</h2>
  {table_html}
  {perf_html}
  {plot_div}
</body></html>
    """

    path = f"{OUT_DIR}/vix_ratio_final.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"最近五日：")
    for r in last5:
        print(f"  {r['date']}  VIX={r['vix']:.2f}  MA3={r['ma3']:.2f}  MA30={r['ma30']:.2f}  "
              f"ratio={r['ratio']:.3f}  -> {r['signal']}")
    print(f"\n全样本收益：VIX比值={mt*100:.1f}% (MDD {mdt*100:.1f}%)  SPY={st*100:.1f}% (MDD {sdt*100:.1f}%)")
    print(f"样本外收益：VIX比值={mo*100:.1f}% (MDD {mdo*100:.1f}%)  SPY={so*100:.1f}% (MDD {sdo*100:.1f}%)")
    print(f"\n交付：{path}")


if __name__ == "__main__":
    main()
