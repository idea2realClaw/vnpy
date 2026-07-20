"""
run_vix_ma_qqq_vxx.py —— VIX_MA3/VIX_MA30 正向择时 QQQ，但「空仓」改为「轮动持 VXX」

合成：ratio = VIX_MA3 / VIX_MA30
方向（正向，沿用策略一 VixMA3MA30Ratio 逻辑）：
    ratio >  thr -> 轮动到 VXX（恐慌期做多波动率）
    ratio <= thr -> 持有 QQQ（100%满仓）
每次进入 VXX 至少连续持有 MIN_VXX_DAYS 天（沿用上一轮「最少5天」约束，只是把现金换成 VXX）。

注意：VXX 数据从 2018-01-25 才有，故本回测窗口为 2018-2026（与 VIX/QQQ 的交集），
显著短于原现金版（2011-2026）。为公平比较，同窗口也给出「现金版」与「QQQ 买入持有」基准。

交付阈值沿用策略一既定 thr=2.0（未对 QQQ/VXX 重拟合，做跨资产验证）。
本文件不修改任何已 tag 的 VVIX / VixMA3MA30Ratio / VixMA3MA30QQQ 文件。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vix_ma_qqq_vxx.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline
from plotly.subplots import make_subplots

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"

# 进入 VXX 后至少连续持有天数（沿用上轮「空仓至少5天」约束，对象换成 VXX）
MIN_VXX_DAYS = 5

# VIX 比值有效上限较高，覆盖 1.3~2.6
THRS = [round(1.3 + 0.1 * i, 2) for i in range(0, 14)]  # 1.30..2.60
DELIVERY_THR = 2.0  # 沿用策略一既定阈值，不重拟合

# 窗口（受 VXX 数据限制，全样本=2018-2026）
FULL0, FULL1 = "2018-01-25", "2026-07-20"
OOS0, OOS1 = "2021-01-01", "2026-07-20"


def enforce_min_vxx(want_vxx, min_days):
    """want_vxx[t]: 原始信号是否要求持 VXX(布尔)。返回 in_vxx 布尔数组，
    并保证任一次进入 VXX 至少连续 min_days 天。"""
    n = len(want_vxx)
    in_vxx = np.zeros(n, dtype=bool)
    cd = 0
    for t in range(n):
        if cd > 0:
            in_vxx[t] = True
            cd -= 1
            continue
        if want_vxx[t]:
            in_vxx[t] = True
            cd = min_days - 1
        else:
            in_vxx[t] = False
    return in_vxx


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


def rotate_nav(ratio, valid, n, ret_q, ret_v, thr):
    """ratio>thr -> 持 VXX，否则持 QQQ；进 VXX 至少 MIN_VXX_DAYS 天。返回 (nav, in_vxx)。"""
    want = np.array([
        (valid[t] and not np.isnan(ratio[t]) and ratio[t] > thr)
        for t in range(n)
    ])
    in_vxx = enforce_min_vxx(want, MIN_VXX_DAYS)
    nav = [100.0]
    for t in range(n):
        r = ret_v[t] if in_vxx[t] else ret_q[t]
        nav.append(nav[-1] * (1.0 + r))
    return np.array(nav), in_vxx


def cash_nav(ratio, valid, n, ret, thr):
    """原现金版（对照）：ratio>thr -> 空仓，否则持 QQQ。返回 (nav, pos)。"""
    want = np.array([
        (valid[t] and not np.isnan(ratio[t]) and ratio[t] > thr)
        for t in range(n)
    ])
    pos = np.ones(n)
    cd = 0
    for t in range(n):
        if cd > 0:
            pos[t] = 0.0
            cd -= 1
            continue
        if want[t]:
            pos[t] = 0.0
            cd = MIN_VXX_DAYS - 1
        else:
            pos[t] = 1.0
    nav = [100.0]
    for t in range(n):
        nav.append(nav[-1] * (1.0 + pos[t] * ret[t]))
    return np.array(nav), pos


def main():
    # ---- 数据 ----
    vix = load_close("VIX")
    qqq = load_close("QQQ")
    vxx = load_close("VXX")
    idx = vix.index.intersection(qqq.index).intersection(vxx.index)
    VIX = vix.loc[idx].astype(float)
    QQQ = qqq.loc[idx].astype(float)
    VXX = vxx.loc[idx].astype(float)
    dates = idx.to_numpy()
    q = QQQ.to_numpy(float)
    x = VXX.to_numpy(float)
    ret_q = np.diff(q) / q[:-1]
    ret_v = np.diff(x) / x[:-1]
    v = VIX.to_numpy(float)

    # ---- 平滑 ----
    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().to_numpy()
    ratio = ma3 / ma30

    n = len(ret_q)
    dt = dates[:n]
    valid = ~np.isnan(ratio[:n]) & (dt >= np.datetime64("2011-01-01")) & (dt >= np.datetime64(FULL0))

    ratio_min, ratio_max = np.nanmin(ratio), np.nanmax(ratio)
    print(f"共同交易日(VIX∩QQQ∩VXX): {len(dates)}  ({dates[0].astype('M8[D]')} ~ {dates[-1].astype('M8[D]')})")
    print(f"ratio(VIX_MA3/VIX_MA30) 范围: {ratio_min:.2f} ~ {ratio_max:.2f}  "
          f"(有效均值 {np.nanmean(ratio[:n][valid]):.3f})")

    # 基准
    qqq_nav = [100.0]
    for t in range(n):
        qqq_nav.append(qqq_nav[-1] * (1.0 + ret_q[t]))
    qqq_nav = np.array(qqq_nav)
    vxx_nav = [100.0]
    for t in range(n):
        vxx_nav.append(vxx_nav[-1] * (1.0 + ret_v[t]))
    vxx_nav = np.array(vxx_nav)
    st, sdt, _ = metrics(qqq_nav, dates, FULL0, FULL1)
    so, sdo, _ = metrics(qqq_nav, dates, OOS0, OOS1)
    xt, xdt, _ = metrics(vxx_nav, dates, FULL0, FULL1)
    print(f"QQQ 买入持有：全样本 {st*100:.1f}%(MDD {sdt*100:.1f}%)  样本外 {so*100:.1f}%(MDD {sdo*100:.1f}%)")
    print(f"VXX 买入持有：全样本 {xt*100:.1f}%(MDD {xdt*100:.1f}%)")

    # ---- 网格扫描（轮动版）----
    scan = []
    for thr in THRS:
        nav, in_vxx = rotate_nav(ratio, valid, n, ret_q, ret_v, thr)
        m = metrics(nav, dates, FULL0, FULL1)
        if m is None:
            continue
        tot, mdd, rd = m
        vxx_days = float(np.mean(in_vxx))
        scan.append((rd, thr, tot, mdd, vxx_days))

    # 交付：优先既定阈值 2.0
    hit = [c for c in scan if abs(c[1] - DELIVERY_THR) < 1e-9]
    DTHR = DELIVERY_THR if hit else max(scan, key=lambda x: x[0])[1]

    nav, in_vxx = rotate_nav(ratio, valid, n, ret_q, ret_v, DTHR)
    cash, _ = cash_nav(ratio, valid, n, ret_q, DTHR)

    # 关键绩效
    rt_, rdt_, _ = metrics(nav, dates, FULL0, FULL1)
    ro_, rdo_, _ = metrics(nav, dates, OOS0, OOS1)
    ct_, cdt_, _ = metrics(cash, dates, FULL0, FULL1)
    co_, cdo_, _ = metrics(cash, dates, OOS0, OOS1)
    vxx_frac = float(np.mean(in_vxx))
    print(f"\n[轮动 VXX] 交付 thr={DTHR:.2f}  VXX持仓占比={vxx_frac*100:.1f}%")
    print(f"  全样本 {rt_*100:.1f}%(MDD {rdt_*100:.1f}%)  样本外 {ro_*100:.1f}%(MDD {rdo_*100:.1f}%)")
    print(f"[同窗口 现金版] 全样本 {ct_*100:.1f}%(MDD {cdt_*100:.1f}%)  样本外 {co_*100:.1f}%(MDD {cdo_*100:.1f}%)")
    print(f"\n扫描（thr 升序，全样本）：")
    for c in sorted(scan, key=lambda x: x[1]):
        print(f"  thr={c[1]:.2f}  tot={c[2]*100:.1f}%  MDD={c[3]*100:.1f}%  rd={c[0]:.2f}  VXX持仓={c[4]*100:.1f}%")

    # ---- 顶部：最近五日 ----
    good = [i for i in range(len(dates)) if not np.isnan(ratio[i])]
    last5 = []
    for j in good[-5:]:
        r = ratio[j]
        sig = "持VXX" if (not np.isnan(r) and r > DTHR) else "持有QQQ"
        last5.append({
            "date": pd.Timestamp(dates[j]).strftime("%Y-%m-%d"),
            "vix": v[j], "ma3": ma3[j], "ma30": ma30[j], "ratio": r, "sig": sig,
        })

    # ---- 净值分段 ----
    def seg(nav_arr, d0, d1):
        m = (dates >= np.datetime64(d0)) & (dates <= np.datetime64(d1))
        ii = np.where(m)[0]
        s = nav_arr[ii[0]:ii[-1] + 1]
        return s / s[0], dates[ii[0]:ii[-1] + 1]

    nav_full, nav_oos = seg(nav, FULL0, FULL1)[0], seg(nav, OOS0, OOS1)[0]
    qqq_full, qqq_oos = seg(qqq_nav, FULL0, FULL1)[0], seg(qqq_nav, OOS0, OOS1)[0]
    cash_full, cash_oos = seg(cash, FULL0, FULL1)[0], seg(cash, OOS0, OOS1)[0]

    # ---- HTML 表 ----
    def fmt(x, d=2):
        return f"{x:.{d}f}"
    rows = "".join(
        f"<tr><td>{r['date']}</td><td>{fmt(r['vix'])}</td><td>{fmt(r['ma3'])}</td>"
        f"<td>{fmt(r['ma30'])}</td><td>{fmt(r['ratio'])}</td>"
        f"<td class='{'vxx' if r['sig']=='持VXX' else 'full'}'>{r['sig']}</td></tr>"
        for r in last5
    )
    table_html = f"""
    <div class='panel'>
      <h3>最近五个交易日 VIX 指标与仓位信号（thr={DTHR:.2f}；高比值→持VXX / 低比值→持QQQ；持VXX至少{MIN_VXX_DAYS}天）</h3>
      <table>
        <thead><tr><th>日期</th><th>VIX</th><th>VIX_MA3</th><th>VIX_MA30</th>
        <th>VIX_MA3 / VIX_MA30</th><th>信号</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class='note'>VXX 数据自 2018-01-25 起，故本回测窗口为 <b>{FULL0} ~ {FULL1}</b>（受数据限制，短于原现金版的 2011-2026）。
      判定：ratio &gt; {DTHR:.2f} → 轮动持 VXX；否则 100% 持 QQQ。每次进 VXX 至少连续 {MIN_VXX_DAYS} 天。</p>
    </div>
    """

    perf_html = f"""
    <div class='panel'>
      <h3>关键绩效（VIX_MA3/VIX_MA30 正向，恐慌期 QQQ↔VXX 轮动；窗口 {FULL0}~{FULL1}）</h3>
      <table>
        <thead><tr><th>窗口</th><th>轮动VXX 收益</th><th>轮动VXX 回撤</th>
        <th>同窗口现金版 收益</th><th>同窗口现金版 回撤</th>
        <th>QQQ 收益</th><th>QQQ 回撤</th></tr></thead>
        <tbody>
          <tr><td>全样本 {FULL0[0:4]}-{FULL1[0:4]}</td><td>{rt_*100:.1f}%</td><td>{rdt_*100:.1f}%</td>
              <td>{ct_*100:.1f}%</td><td>{cdt_*100:.1f}%</td>
              <td>{st*100:.1f}%</td><td>{sdt*100:.1f}%</td></tr>
          <tr><td>样本外 {OOS0[0:4]}-{OOS1[0:4]}</td><td>{ro_*100:.1f}%</td><td>{rdo_*100:.1f}%</td>
              <td>{co_*100:.1f}%</td><td>{cdo_*100:.1f}%</td>
              <td>{so*100:.1f}%</td><td>{sdo*100:.1f}%</td></tr>
        </tbody>
      </table>
      <p class='note'>VXX 卖出持有本身长期大幅亏损（全样本 {xt*100:.1f}%，MDD {xdt*100:.1f}%），因波动率期货展期损耗。
      轮动策略仅在 VIX 比值突破 {DTHR:.2f} 的极端恐慌段短暂持有 VXX，意图捕获恐慌尖峰。VXX 持仓占比约 {vxx_frac*100:.1f}%。
      与<b>同窗口现金版</b>对照可见：恐慌期「反手做多 VXX」远胜「空仓」(全样本 {rt_*100:.1f}% vs {ct_*100:.1f}%)，因 COVID 等尖峰中 VXX 暴涨而现金仅避险。
      <b>但须高度警惕：样本外 2021-2026（含 2022 熊市）比值从未&gt;{DTHR:.2f}，三种策略收益完全相同（{ro_*100:.1f}%），即轮动 VXX 在该段等于纯 QQQ 买入持有——全部超额收益来自 2018-2020、尤其是 2020-03 COVID 单次事件，属尾部集中、样本内驱动，并非稳健择时。</b></p>
    </div>
    """

    def scan_rows(res, thr):
        return "".join(
            f"<tr><td>{c[1]:.2f}</td><td>{c[2]*100:.1f}%</td><td>{c[3]*100:.1f}%</td>"
            f"<td>{c[0]:.2f}</td><td>{c[4]*100:.1f}%</td>"
            f"{'<td class=\"warn\">← 交付采用</td>' if abs(c[1]-thr)<1e-9 else '<td></td>'}</tr>"
            for c in sorted(res, key=lambda x: x[1])
        )
    scan_html = f"""
    <div class='panel'>
      <h3>轮动版阈值扫描（全样本 {FULL0}~{FULL1}，高比值→持VXX）</h3>
      <table><thead><tr><th>阈值 thr</th><th>收益</th><th>最大回撤</th><th>收益/回撤</th><th>VXX持仓占比</th><th>备注</th></tr></thead>
      <tbody>{scan_rows(scan, DTHR)}</tbody></table>
    </div>"""

    # ---- 图表 ----
    def two_panel(nav_full, nav_oos, qqq_full, qqq_oos, cash_full, cash_oos, title):
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            subplot_titles=(f"全样本 {FULL0}~{FULL1}（起点=100）", f"样本外 {OOS0}~{OOS1}（起点=100）"),
            vertical_spacing=0.12, row_heights=[0.5, 0.5],
        )
        fig.add_trace(go.Scatter(x=seg(nav_full, FULL0, FULL1)[1], y=nav_full,
                                 name="轮动VXX", line=dict(color="#d62728", width=2.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=seg(qqq_full, FULL0, FULL1)[1], y=qqq_full,
                                 name="QQQ 买入持有", line=dict(color="#1f77b4", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=seg(cash_full, FULL0, FULL1)[1], y=cash_full,
                                 name="同窗口现金版", line=dict(color="#7f7f7f", width=1.6, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=seg(nav_oos, OOS0, OOS1)[1], y=nav_oos,
                                 name="轮动VXX", line=dict(color="#d62728", width=2.4), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=seg(qqq_oos, OOS0, OOS1)[1], y=qqq_oos,
                                 name="QQQ 买入持有", line=dict(color="#1f77b4", width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=seg(cash_oos, OOS0, OOS1)[1], y=cash_oos,
                                 name="同窗口现金版", line=dict(color="#7f7f7f", width=1.6, dash="dot"), showlegend=False), row=2, col=1)
        fig.update_layout(title=title, template="plotly_white", height=860,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="left", x=0),
                          margin=dict(t=70, b=70))
        fig.update_yaxes(title_text="净值（起点=100）", row=1, col=1)
        fig.update_yaxes(title_text="净值（起点=100）", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)
        return plot_offline(fig, include_plotlyjs="cdn", output_type="div", auto_open=False)

    div = two_panel(nav_full, nav_oos, qqq_full, qqq_oos, cash_full, cash_oos,
                    f"VIX_MA3/VIX_MA30 正向：恐慌期 QQQ↔VXX 轮动（thr={DTHR:.2f}，持VXX至少{MIN_VXX_DAYS}天）vs QQQ / 现金版")

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>VIX比值择时QQQ轮动VXX</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; color:#222; }}
  h2 {{ color:#1f2d5a; }}
  .panel {{ background:#fafbfc; border:1px solid #e5e8ee; border-radius:10px; padding:14px 18px; margin:14px 0; }}
  table {{ border-collapse: collapse; width:100%; font-size:14px; }}
  th, td {{ border:1px solid #e1e5ec; padding:8px 10px; text-align:center; }}
  th {{ background:#eef2f8; color:#33415c; }}
  tbody tr:nth-child(even) {{ background:#f6f8fb; }}
  td.vxx {{ color:#b9770e; font-weight:700; }}
  td.full {{ color:#1e8449; font-weight:700; }}
  td.warn {{ color:#b9770e; font-weight:700; }}
  .note {{ color:#6b7280; font-size:12px; margin:8px 0 0; }}
</style></head>
<body>
  <h2>VIX_MA3 / VIX_MA30 正向择时 QQQ —— 恐慌期轮动持 VXX（持VXX至少{MIN_VXX_DAYS}天）vs QQQ / 现金版</h2>
  {table_html}
  {perf_html}
  {scan_html}
  {div}
</body></html>
    """
    path = f"{OUT_DIR}/vix_ma_qqq_vxx.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"\n最近五日：")
    for r in last5:
        print(f"  {r['date']}  VIX={r['vix']:.2f}  MA3={r['ma3']:.2f}  MA30={r['ma30']:.2f}  "
              f"ratio={r['ratio']:.3f}  -> {r['sig']}")
    print(f"\n[结果-全样本] 轮动VXX {rt_*100:.1f}%(MDD {rdt_*100:.1f}%)  现金版 {ct_*100:.1f}%(MDD {cdt_*100:.1f}%)  QQQ {st*100:.1f}%(MDD {sdt*100:.1f}%)")
    print(f"[结果-样本外] 轮动VXX {ro_*100:.1f}%(MDD {rdo_*100:.1f}%)  现金版 {co_*100:.1f}%(MDD {cdo_*100:.1f}%)  QQQ {so*100:.1f}%(MDD {sdo*100:.1f}%)")
    print(f"\n交付：{path}")


if __name__ == "__main__":
    main()
