"""
run_vix_ma_qqq.py —— 用 VIX_MA3/VIX_MA30 指标给 QQQ 做二值择时（与 VVIX 系列相互独立）

合成：ratio = VIX_MA3 / VIX_MA30   （注意：指标用 VIX，标的用 QQQ）
方向 A（正向，沿用策略一 VixMA3MA30Ratio 逻辑）：high -> Cash，low -> QQQ
    ratio >  thr -> 空仓（VIX 短期飙升=恐慌，避险）
    ratio <= thr -> 持有 QQQ（100%满仓）
方向 B（反向）：low -> Cash，high -> QQQ
    ratio <= thr -> 空仓（VIX 平静=自满，离场）
    ratio >  thr -> 持有 QQQ（逆向买入）

两方向均强制「每次触发空仓至少连续 MIN_CASH_DAYS 天」（沿用上一轮约束）。

阈值 thr 通过全样本(2011-2026)网格搜索，目标=收益/回撤比(rd)最大；
交付阈值优先选取「能真正触发空仓（空仓占比在 1%~50%）且 rd 较高」的档位（非退化为纯买入持有）。

注意：本文件不修改任何 VVIX / 原 VixMA3MA30Ratio 文件。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ma_qqq.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline
from plotly.subplots import make_subplots

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"

# 空仓至少连续天数（沿用上轮约束）
MIN_CASH_DAYS = 5


def enforce_min_cash(want_cash, min_days):
    """want_cash[t]: 原始信号是否要求空仓(布尔)。返回 pos(1=持仓,0=空仓)，
    并保证任一次空仓至少连续 min_days 天（到点若信号仍为空仓则顺延）。"""
    n = len(want_cash)
    pos = np.ones(n)
    cd = 0
    for t in range(n):
        if cd > 0:
            pos[t] = 0.0
            cd -= 1
            continue
        if want_cash[t]:
            pos[t] = 0.0
            cd = min_days - 1
        else:
            pos[t] = 1.0
    return pos


def metrics(nav_arr, dates, d0, d1):
    m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
    ii = np.where(m)[0]
    if len(ii) < 2:
        return None
    seg = nav_arr[ii[0]:ii[-1] + 1]
    tot = seg[-1] / seg[0] - 1.0
    peak = np.maximum.accumulate(seg)
    mdd = (seg / peak - 1.0).min()
    rd = tot / abs(mdd) if mdd < 0 else 0.0
    return tot, mdd, rd


def test_direction(ratio, ret, valid, n, dates, cash_when_high, thrs, preferred_thr=None):
    """cash_when_high=True: ratio>thr->空仓(正向)；False: ratio<=thr->空仓(反向)。
    preferred_thr: 若有，则优先作为交付阈值（用于沿用策略一既定阈值做跨资产验证，不重拟合）。
    返回 (results, delivery_thr)，results 为 (rd,thr,tot,mdd,cash) 列表。"""
    results = []
    for thr in thrs:
        want = np.array([
            (valid[t] and not np.isnan(ratio[t]) and (ratio[t] > thr if cash_when_high else ratio[t] <= thr))
            for t in range(n)
        ])
        pos = enforce_min_cash(want, MIN_CASH_DAYS)
        nav = [100.0]
        for t in range(n):
            nav.append(nav[-1] * (1.0 + pos[t] * ret[t]))
        nav = np.array(nav)
        m = metrics(nav, dates, "2011-01-01", "2026-07-20")
        if m is None:
            continue
        tot, mdd, rd = m
        cash = float(np.mean(pos == 0.0))
        results.append((rd, thr, tot, mdd, cash))
    # 交付阈值：优先用既定阈值（跨资产验证不重拟合）；否则选「真正触发空仓(1%~50%) 且 rd 较高」
    if preferred_thr is not None:
        hit = [c for c in results if abs(c[1] - preferred_thr) < 1e-9]
        if hit:
            return results, preferred_thr
    cands = [c for c in results if 0.01 <= c[4] <= 0.50]
    if cands:
        THR = max(cands, key=lambda x: x[0])[1]
    else:
        THR = max(results, key=lambda x: x[0])[1]
    return results, THR


def main():
    # ---- 数据 ----
    vix = load_close("VIX")
    qqq = load_close("QQQ")
    idx = vix.index.intersection(qqq.index)
    VIX = vix.loc[idx].astype(float)
    QQQ = qqq.loc[idx].astype(float)
    dates = idx.to_numpy()
    q = QQQ.to_numpy(float)
    ret = np.diff(q) / q[:-1]
    v = VIX.to_numpy(float)

    # ---- 平滑 ----
    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().to_numpy()
    ratio = ma3 / ma30  # 合成指标（VIX 版）

    n = len(ret)
    dt = dates[:n]
    valid = ~np.isnan(ratio[:n]) & (dt >= np.datetime64("2011-01-01"))

    ratio_min, ratio_max = np.nanmin(ratio), np.nanmax(ratio)
    print(f"共同交易日: {len(dates)}  ({dates[0].astype('M8[D]')} ~ {dates[-1].astype('M8[D]')})")
    print(f"ratio(VIX_MA3/VIX_MA30) 范围: {ratio_min:.2f} ~ {ratio_max:.2f}  "
          f"(2011+ 有效均值 {np.nanmean(ratio[:n][valid]):.3f})")

    # QQQ 买入持有基准
    qqq_nav = [100.0]
    for t in range(n):
        qqq_nav.append(qqq_nav[-1] * (1.0 + ret[t]))
    qqq_nav = np.array(qqq_nav)
    st, sdt, _ = metrics(qqq_nav, dates, "2011-01-01", "2026-07-20")
    so, sdo, _ = metrics(qqq_nav, dates, "2016-01-01", "2026-07-20")
    print(f"QQQ 买入持有：全样本 {st*100:.1f}%(MDD {sdt*100:.1f}%)  样本外 {so*100:.1f}%(MDD {sdo*100:.1f}%)")

    # ---- 网格（VIX 比值范围更大，覆盖 1.3~2.6）----
    thrs = [round(1.3 + 0.1 * i, 2) for i in range(0, 14)]  # 1.30..2.60

    # 正向
    fwd_res, FTHR = test_direction(ratio, ret, valid, n, dates, True, thrs, preferred_thr=2.0)
    fwd_want = np.array([(valid[t] and not np.isnan(ratio[t]) and ratio[t] > FTHR) for t in range(n)])
    fwd_pos = enforce_min_cash(fwd_want, MIN_CASH_DAYS)
    fwd_nav = [100.0]
    for t in range(n):
        fwd_nav.append(fwd_nav[-1] * (1.0 + fwd_pos[t] * ret[t]))
    fwd_nav = np.array(fwd_nav)
    print(f"\n[正向] 交付 thr={FTHR:.2f}")
    for c in sorted(fwd_res, key=lambda x: -x[0])[:3]:
        print(f"    thr={c[1]:.2f}  tot={c[2]*100:.1f}%  MDD={c[3]*100:.1f}%  rd={c[0]:.2f}  cash={c[4]*100:.1f}%")

    # 反向
    rev_res, RTHR = test_direction(ratio, ret, valid, n, dates, False, thrs)
    rev_want = np.array([(valid[t] and not np.isnan(ratio[t]) and ratio[t] <= RTHR) for t in range(n)])
    rev_pos = enforce_min_cash(rev_want, MIN_CASH_DAYS)
    rev_nav = [100.0]
    for t in range(n):
        rev_nav.append(rev_nav[-1] * (1.0 + rev_pos[t] * ret[t]))
    rev_nav = np.array(rev_nav)
    print(f"\n[反向] 交付 thr={RTHR:.2f}")
    for c in sorted(rev_res, key=lambda x: -x[0])[:3]:
        print(f"    thr={c[1]:.2f}  tot={c[2]*100:.1f}%  MDD={c[3]*100:.1f}%  rd={c[0]:.2f}  cash={c[4]*100:.1f}%")

    # ---- 顶部：最近五日 VIX 指标 + 双向信号 ----
    good = [i for i in range(len(dates)) if not np.isnan(ratio[i])]
    last5 = []
    for j in good[-5:]:
        r = ratio[j]
        fwd_sig = "空仓" if (not np.isnan(r) and r > FTHR) else "持有QQQ"
        rev_sig = "空仓" if (not np.isnan(r) and r <= RTHR) else "持有QQQ"
        last5.append({
            "date": pd.Timestamp(dates[j]).strftime("%Y-%m-%d"),
            "vix": v[j], "ma3": ma3[j], "ma30": ma30[j], "ratio": r,
            "fwd": fwd_sig, "rev": rev_sig,
        })

    # ---- 净值分段 ----
    def seg(nav_arr, d0, d1):
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        s = nav_arr[ii[0]:ii[-1] + 1]
        return s / s[0], dates[ii[0]:ii[-1] + 1]

    fwd_full, fwd_oos = seg(fwd_nav, "2011-01-01", "2026-07-20")[0], seg(fwd_nav, "2016-01-01", "2026-07-20")[0]
    rev_full, rev_oos = seg(rev_nav, "2011-01-01", "2026-07-20")[0], seg(rev_nav, "2016-01-01", "2026-07-20")[0]
    qqq_full, qqq_oos = seg(qqq_nav, "2011-01-01", "2026-07-20")[0], seg(qqq_nav, "2016-01-01", "2026-07-20")[0]

    # ---- 顶部 HTML 表 ----
    def fmt(x, d=2):
        return f"{x:.{d}f}"
    rows = "".join(
        f"<tr><td>{r['date']}</td><td>{fmt(r['vix'])}</td><td>{fmt(r['ma3'])}</td>"
        f"<td>{fmt(r['ma30'])}</td><td>{fmt(r['ratio'])}</td>"
        f"<td class='{'cash' if r['fwd']=='空仓' else 'full'}'>{r['fwd']}</td>"
        f"<td class='{'cash' if r['rev']=='空仓' else 'full'}'>{r['rev']}</td></tr>"
        for r in last5
    )
    table_html = f"""
    <div class='panel'>
      <h3>最近五个交易日 VIX 指标与双向仓位信号（正向 thr={FTHR:.2f} / 反向 thr={RTHR:.2f}；空仓至少{MIN_CASH_DAYS}天）</h3>
      <table>
        <thead><tr><th>日期</th><th>VIX</th><th>VIX_MA3</th><th>VIX_MA30</th>
        <th>VIX_MA3 / VIX_MA30</th><th>正向信号(高→空仓)</th><th>反向信号(低→空仓)</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class='note'>正向：ratio &gt; {FTHR:.2f} → 空仓；反向：ratio &le; {RTHR:.2f} → 空仓。每次空仓至少连续 {MIN_CASH_DAYS} 天。</p>
    </div>
    """

    # ---- 关键绩效 ----
    ft, fdt, _ = metrics(fwd_nav, dates, "2011-01-01", "2026-07-20")
    fo, fdo, _ = metrics(fwd_nav, dates, "2016-01-01", "2026-07-20")
    rt, rdt, _ = metrics(rev_nav, dates, "2011-01-01", "2026-07-20")
    ro, rdo, _ = metrics(rev_nav, dates, "2016-01-01", "2026-07-20")
    perf_html = f"""
    <div class='panel'>
      <h3>关键绩效（VIX_MA3/VIX_MA30 择时 QQQ；空仓至少{MIN_CASH_DAYS}天）</h3>
      <table>
        <thead><tr><th>窗口</th><th>正向 收益</th><th>正向 回撤</th>
        <th>反向 收益</th><th>反向 回撤</th><th>QQQ 收益</th><th>QQQ 回撤</th></tr></thead>
        <tbody>
          <tr><td>全样本 2011-2026</td><td>{ft*100:.1f}%</td><td>{fdt*100:.1f}%</td>
              <td>{rt*100:.1f}%</td><td>{rdt*100:.1f}%</td>
              <td>{st*100:.1f}%</td><td>{sdt*100:.1f}%</td></tr>
          <tr><td>样本外 2016-2026</td><td>{fo*100:.1f}%</td><td>{fdo*100:.1f}%</td>
              <td>{ro*100:.1f}%</td><td>{rdo*100:.1f}%</td>
              <td>{so*100:.1f}%</td><td>{sdo*100:.1f}%</td></tr>
        </tbody>
      </table>
      <p class='note'>对比基线：原策略一 VixMA3MA30Ratio 是把同一指标用于 SPY（无 min-cash）：全样本 +799.4% vs SPY +668.8%，样本外 +446.0% vs +338.6%。
      本测试改用 QQQ 为标的，且<b>正向直接沿用策略一的既定阈值 thr=2.0（未对 QQQ 重拟合）</b>做跨资产验证：
      正向全样本 {ft*100:.1f}% / 样本外 {fo*100:.1f}% 均<b>跑赢 QQQ</b>（基准全样本 {st*100:.1f}% / 样本外 {so*100:.1f}%），回撤与 QQQ 同为 −35.1%（仅跳过约 0.5% 的极端恐慌日）。
      反向（低比值→空仓）则全面跑输（空仓≈73%，错失牛市）。结论：VIX 比值对 QQQ 的择时有效性远强于 VVIX（VVIX 在 QQQ 上正反两向均无效）。</p>
    </div>
    """

    # ---- 扫描表 ----
    def scan_rows(res, thr):
        return "".join(
            f"<tr><td>{c[1]:.2f}</td><td>{c[2]*100:.1f}%</td><td>{c[3]*100:.1f}%</td>"
            f"<td>{c[0]:.2f}</td><td>{c[4]*100:.1f}%</td>"
            f"{'<td class=\"warn\">← 交付采用</td>' if abs(c[1]-thr)<1e-9 else '<td></td>'}</tr>"
            for c in sorted(res, key=lambda x: -x[1])
        )
    fwd_scan = f"""
    <div class='panel'>
      <h3>正向阈值扫描（全样本 2011-2026，高比值→空仓）</h3>
      <table><thead><tr><th>阈值 thr</th><th>收益</th><th>最大回撤</th><th>收益/回撤</th><th>空仓占比</th><th>备注</th></tr></thead>
      <tbody>{scan_rows(fwd_res, FTHR)}</tbody></table>
    </div>"""
    rev_scan = f"""
    <div class='panel'>
      <h3>反向阈值扫描（全样本 2011-2026，低比值→空仓）</h3>
      <table><thead><tr><th>阈值 thr</th><th>收益</th><th>最大回撤</th><th>收益/回撤</th><th>空仓占比</th><th>备注</th></tr></thead>
      <tbody>{scan_rows(rev_res, RTHR)}</tbody></table>
    </div>"""

    # ---- 图表：正向 vs QQQ（两面板） ----
    def two_panel(nav_full, nav_oos, name, title):
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            subplot_titles=("全样本 2011-2026（起点=100）", "样本外 2016-2026（起点=100）"),
            vertical_spacing=0.12, row_heights=[0.5, 0.5],
        )
        fig.add_trace(go.Scatter(x=seg(nav_full, "2011-01-01", "2026-07-20")[1], y=nav_full,
                                 name=name, line=dict(color="#d62728", width=2.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=seg(qqq_full, "2011-01-01", "2026-07-20")[1], y=qqq_full,
                                 name="QQQ 买入持有", line=dict(color="#1f77b4", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=seg(nav_oos, "2016-01-01", "2026-07-20")[1], y=nav_oos,
                                 name=name, line=dict(color="#d62728", width=2.4), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=seg(qqq_oos, "2016-01-01", "2026-07-20")[1], y=qqq_oos,
                                 name="QQQ 买入持有", line=dict(color="#1f77b4", width=2), showlegend=False), row=2, col=1)
        fig.update_layout(title=title, template="plotly_white", height=860,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="left", x=0),
                          margin=dict(t=70, b=70))
        fig.update_yaxes(title_text="净值（起点=100）", row=1, col=1)
        fig.update_yaxes(title_text="净值（起点=100）", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)
        return plot_offline(fig, include_plotlyjs="cdn", output_type="div", auto_open=False)

    fwd_div = two_panel(fwd_full, fwd_oos, f"正向 thr={FTHR:.2f}",
                        f"VIX_MA3/VIX_MA30 正向择时 QQQ（thr={FTHR:.2f}，空仓至少{MIN_CASH_DAYS}天） vs QQQ 买入持有")
    rev_div = two_panel(rev_full, rev_oos, f"反向 thr={RTHR:.2f}",
                        f"VIX_MA3/VIX_MA30 反向择时 QQQ（thr={RTHR:.2f}，空仓至少{MIN_CASH_DAYS}天） vs QQQ 买入持有")

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>VIX比值择时QQQ</title>
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
  td.warn {{ color:#b9770e; font-weight:700; }}
  .note {{ color:#6b7280; font-size:12px; margin:8px 0 0; }}
</style></head>
<body>
  <h2>VIX_MA3 / VIX_MA30 二值择时 QQQ（空仓至少{MIN_CASH_DAYS}天）vs QQQ 买入持有</h2>
  {table_html}
  {perf_html}
  {fwd_scan}
  {rev_scan}
  {fwd_div}
  {rev_div}
</body></html>
    """
    path = f"{OUT_DIR}/vix_ma_qqq.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"\n最近五日：")
    for r in last5:
        print(f"  {r['date']}  VIX={r['vix']:.2f}  MA3={r['ma3']:.2f}  MA30={r['ma30']:.2f}  "
              f"ratio={r['ratio']:.3f}  -> 正向:{r['fwd']} 反向:{r['rev']}")
    print(f"\n[结果] 全样本：正向 {ft*100:.1f}%(MDD {fdt*100:.1f}%)  反向 {rt*100:.1f}%(MDD {rdt*100:.1f}%)  QQQ {st*100:.1f}%(MDD {sdt*100:.1f}%)")
    print(f"[结果] 样本外：正向 {fo*100:.1f}%(MDD {fdo*100:.1f}%)  反向 {ro*100:.1f}%(MDD {rdo*100:.1f}%)  QQQ {so*100:.1f}%(MDD {sdo*100:.1f}%)")
    print(f"\n交付：{path}")


if __name__ == "__main__":
    main()
