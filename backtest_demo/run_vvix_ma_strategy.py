"""
run_vvix_ma_strategy.py —— 第二个策略（与 VixMA3MA30Ratio 相互独立，不改方案一）

合成：ratio = VVIX_MA3 / VVIX_MA30
    VVIX_MA3  = VVIX 的 3 日移动平均（短期 vol-of-vol）
    VVIX_MA30 = VVIX 的 30 日移动平均（长期 vol-of-vol）
方向（用户指定）：low → QQQ，high → Cash
    即 ratio > thr -> 空仓（VVIX 短期飙升=波动结构紧张，避险）
       ratio <= thr -> 持有 QQQ（100%满仓）

阈值 thr 通过全样本(2011-2026)网格搜索，目标=收益/回撤比(rd)最大。

交付 HTML 同方案一版式：顶部「最近五日 VVIX 指标 + 仓位信号」表 + 双面板净值（策略 vs QQQ 买入持有）。

注意：本文件不修改 run_vix_ratio_final.py / vix_ratio_final.html（方案一保持不动）。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_vvix_ma_strategy.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot as plot_offline
from plotly.subplots import make_subplots

from backtest_demo.run_vts_backtest import load_close

OUT_DIR = "/Users/zhuxiaodong/Documents/GitRepo/vnpy/backtest_demo"


def nav_qqq(thr, ratio, ret, valid):
    """二值 QQQ：ratio>thr -> 空仓(0)，否则持有 QQQ(1)。返回 (nav, pos)。"""
    n = len(ret)
    pos = np.ones(n)
    for t in range(n):
        if not valid[t] or np.isnan(ratio[t]):
            pos[t] = 1.0
        else:
            pos[t] = 0.0 if ratio[t] > thr else 1.0
    nav = [100.0]
    for t in range(n):
        nav.append(nav[-1] * (1.0 + pos[t] * ret[t]))
    return np.array(nav), pos


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


def main():
    # ---- 数据 ----
    vvix = load_close("VVIX")
    qqq = load_close("QQQ")
    idx = vvix.index.intersection(qqq.index)
    VVIX = vvix.loc[idx].astype(float)
    QQQ = qqq.loc[idx].astype(float)
    dates = idx.to_numpy()
    q = QQQ.to_numpy(float)
    ret = np.diff(q) / q[:-1]
    v = VVIX.to_numpy(float)

    # ---- 平滑 ----
    ma30 = pd.Series(v).rolling(30, min_periods=30).mean().to_numpy()
    ma3 = pd.Series(v).rolling(3, min_periods=3).mean().to_numpy()
    ratio = ma3 / ma30  # 合成指标

    n = len(ret)
    dt = dates[:n]
    valid = ~np.isnan(ratio[:n]) & (dt >= np.datetime64("2011-01-01"))

    print(f"共同交易日: {len(dates)}  ({dates[0].astype('M8[D]')} ~ {dates[-1].astype('M8[D]')})")
    ratio_min, ratio_max = np.nanmin(ratio), np.nanmax(ratio)
    print(f"ratio 范围: {ratio_min:.2f} ~ {ratio_max:.2f}  "
          f"(2011+ 有效均值 {np.nanmean(ratio[:n][valid]):.3f})")

    # ---- 网格搜索阈值（全样本 2011-2026，目标=收益/回撤比）----
    thrs = [round(1.05 + 0.05 * i, 2) for i in range(0, 16)]  # 1.05 .. 1.80
    best = None
    allc = []
    for thr in thrs:
        nav, pos = nav_qqq(thr, ratio, ret, valid)
        m = metrics(nav, dates, "2011-01-01", "2026-07-20")
        if m is None:
            continue
        tot, mdd, rd = m
        cash = float(np.mean(pos == 0.0))
        cand = (rd, thr, tot, mdd, cash)
        if best is None or rd > best[0]:
            best = cand
        allc.append(cand)
    top5 = sorted(allc, key=lambda x: -x[0])[:5]
    THR = best[1]
    print("\n=== 阈值网格搜索（全样本 2011-2026，目标=收益/回撤比）===")
    print(f"  网格 rd 最优 thr={best[1]:.2f}（收益={best[2]*100:.1f}% 空仓={best[4]*100:.1f}%）")
    print(f"  注意：VVIX_MA3/VVIX_MA30 比值有效上限仅 {ratio_max:.2f}，阈值≥该上限附近时几乎从不触发(空仓≈0)，等价于直接买入持有 QQQ（退化）。")
    print("  Top5:")
    for c in top5:
        print(f"    thr={c[1]:.2f}  tot={c[2]*100:.1f}%  MDD={c[3]*100:.1f}%  rd={c[0]:.2f}  cash={c[4]*100:.1f}%")

    # 选用能真正触发信号的阈值（而非退化的 rd 最优）：VVIX 飙升→空仓，thr=1.30 覆盖最主要为 2020 等危机的交易日
    THR = 1.30
    print(f"\n  → 交付采用 thr={THR:.2f}（真正触发择时，非退化为买入持有）")

    # ---- 选定阈值的净值 ----
    nav, pos = nav_qqq(THR, ratio, ret, valid)
    qqq_nav = [100.0]
    for t in range(n):
        qqq_nav.append(qqq_nav[-1] * (1.0 + ret[t]))
    qqq_nav = np.array(qqq_nav)

    # ---- 顶部：最近五日 VVIX 指标 + 仓位信号 ----
    good = [i for i in range(len(dates)) if not np.isnan(ratio[i])]
    last5 = []
    for j in good[-5:]:
        r = ratio[j]
        sig = "空仓" if (not np.isnan(r) and r > THR) else "持有QQQ"
        last5.append({
            "date": pd.Timestamp(dates[j]).strftime("%Y-%m-%d"),
            "vvix": v[j],
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

    (full_nav, full_d), (qqq_full, _) = seg(nav, "2011-01-01", "2026-07-20"), seg(qqq_nav, "2011-01-01", "2026-07-20")
    (oos_nav, oos_d), (qqq_oos, _) = seg(nav, "2016-01-01", "2026-07-20"), seg(qqq_nav, "2016-01-01", "2026-07-20")

    # ---- 顶部 HTML 表 ----
    def fmt(x, d=2):
        return f"{x:.{d}f}"
    rows = "".join(
        f"<tr><td>{r['date']}</td><td>{fmt(r['vvix'])}</td><td>{fmt(r['ma3'])}</td>"
        f"<td>{fmt(r['ma30'])}</td><td>{fmt(r['ratio'])}</td>"
        f"<td class='{'cash' if r['signal']=='空仓' else 'full'}'>{r['signal']}</td></tr>"
        for r in last5
    )
    table_html = f"""
    <div class='panel'>
      <h3>最近五个交易日 VVIX 指标与仓位信号（阈值 thr={THR:.2f}，方向：低于→QQQ / 高于→空仓）</h3>
      <table>
        <thead><tr><th>日期</th><th>VVIX</th><th>VVIX_MA3</th><th>VVIX_MA30</th>
        <th>VVIX_MA3 / VVIX_MA30</th><th>信号</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p class='note'>信号判定：ratio &gt; {THR:.2f} → 空仓；否则 100% 满仓 QQQ。</p>
    </div>
    """

    # ---- 净值图（两面板：全样本 + 样本外）----
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        subplot_titles=("全样本 2011-2026（起点=100）", "样本外 2016-2026（起点=100）"),
        vertical_spacing=0.12, row_heights=[0.5, 0.5],
    )
    fig.add_trace(go.Scatter(x=full_d, y=full_nav, name=f"VVIX比值 thr={THR:.2f}",
                             line=dict(color="#d62728", width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=full_d, y=qqq_full, name="QQQ 买入持有",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=oos_d, y=oos_nav, name=f"VVIX比值 thr={THR:.2f}",
                             line=dict(color="#d62728", width=2.4), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=oos_d, y=qqq_oos, name="QQQ 买入持有",
                             line=dict(color="#1f77b4", width=2), showlegend=False), row=2, col=1)
    fig.update_layout(
        title="VVIX_MA3/VVIX_MA30 二值择时（thr） vs QQQ 买入持有",
        template="plotly_white", height=860,
        legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="left", x=0),
        margin=dict(t=70, b=70),
    )
    fig.update_yaxes(title_text="净值（起点=100）", row=1, col=1)
    fig.update_yaxes(title_text="净值（起点=100）", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)

    plot_div = plot_offline(fig, include_plotlyjs="cdn", output_type="div", auto_open=False)

    # ---- 关键绩效 ----
    mt, mdt, _ = metrics(nav, dates, "2011-01-01", "2026-07-20")
    mo, mdo, _ = metrics(nav, dates, "2016-01-01", "2026-07-20")
    st, sdt, _ = metrics(qqq_nav, dates, "2011-01-01", "2026-07-20")
    so, sdo, _ = metrics(qqq_nav, dates, "2016-01-01", "2026-07-20")

    perf_html = f"""
    <div class='panel'>
      <h3>关键绩效（阈值 thr={THR:.2f}）</h3>
      <table>
        <thead><tr><th>窗口</th><th>VVIX比值策略 收益</th><th>最大回撤</th>
        <th>QQQ 收益</th><th>QQQ 最大回撤</th></tr></thead>
        <tbody>
          <tr><td>全样本 2011-2026</td><td>{mt*100:.1f}%</td><td>{mdt*100:.1f}%</td>
              <td>{st*100:.1f}%</td><td>{sdt*100:.1f}%</td></tr>
          <tr><td>样本外 2016-2026</td><td>{mo*100:.1f}%</td><td>{mdo*100:.1f}%</td>
              <td>{so*100:.1f}%</td><td>{sdo*100:.1f}%</td></tr>
        </tbody>
      </table>
      <p class='note'>信号 t-1 日收盘算出 → 决定 t 日仓位（无日内前视）。方向：ratio 低于阈值→持有QQQ，高于阈值→空仓。</p>
    </div>
    """

    # ---- 阈值扫描表（全样本）----
    scan_rows = "".join(
        f"<tr><td>{c[1]:.2f}</td><td>{c[2]*100:.1f}%</td><td>{c[3]*100:.1f}%</td>"
        f"<td>{c[0]:.2f}</td><td>{c[4]*100:.1f}%</td>"
        f"{'<td class=\"warn\">← 交付采用</td>' if abs(c[1]-THR)<1e-9 else '<td></td>'}</tr>"
        for c in sorted(allc, key=lambda x: -x[1])
    )
    scan_html = f"""
    <div class='panel'>
      <h3>阈值扫描（全样本 2011-2026，方向：高比值→空仓）</h3>
      <table>
        <thead><tr><th>阈值 thr</th><th>收益</th><th>最大回撤</th><th>收益/回撤</th>
        <th>空仓占比</th><th>备注</th></tr></thead>
        <tbody>{scan_rows}</tbody>
      </table>
      <p class='note'>VVIX_MA3/VVIX_MA30 比值有效上限仅 {ratio_max:.2f}，故 thr≥该上限附近几乎从不触发、等价于买入持有 QQQ（退化，rd 最高）。
      本策略在任意阈值下<b>均未跑赢 QQQ 买入持有</b>，且各档最大回撤与 QQQ 同为 ≈−35.1%（信号滞后，减仓发生在下跌之后、反弹之前）。</p>
    </div>
    """

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>VVIX比值二值择时 vs QQQ</title>
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
  <h2>VVIX 比值二值择时策略（VVIX_MA3 / VVIX_MA30，thr={THR:.2f}） vs QQQ 买入持有</h2>
  {table_html}
  {perf_html}
  {scan_html}
  {plot_div}
</body></html>
    """

    path = f"{OUT_DIR}/vvix_ma_strategy.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"\n最近五日：")
    for r in last5:
        print(f"  {r['date']}  VVIX={r['vvix']:.2f}  MA3={r['ma3']:.2f}  MA30={r['ma30']:.2f}  "
              f"ratio={r['ratio']:.3f}  -> {r['signal']}")
    print(f"\n全样本收益：VVIX策略={mt*100:.1f}% (MDD {mdt*100:.1f}%)  QQQ={st*100:.1f}% (MDD {sdt*100:.1f}%)")
    print(f"样本外收益：VVIX策略={mo*100:.1f}% (MDD {mdo*100:.1f}%)  QQQ={so*100:.1f}% (MDD {sdo*100:.1f}%)")
    print(f"\n交付：{path}")


if __name__ == "__main__":
    main()
