"""凯利森林算法 (Kelly Forest) —— 固定权重 · 多标的样本外回测 · 单图汇总。

算法 = 冻结随机森林（在恒生指数 HSI 2016–2021 上一次性训练，固定权重，
        运行期不再重训）
      + 凯利百分比仓位  target% = kelly_scale * (p_up - (1 - p_up)) = kelly_scale*(2p-1)
      + 追踪止损。
特征全部无量纲（滞后收益、均线比、动量、波动、RSI、距60日高低），
因此一个在 HSI 上训好的模型可迁移到其它标的做“跨标的样本外推理”。

无未来函数：训练只用 HSI 2016–2021；测试统一为 2022-01-01 ~ 今，
对全部标的（HSI/沪深300/标普500/纳指100/黄金/TLT国债/油气XOP）
均为样本外，训练与回测数据 100% 不重叠。

输出：一张 plotly 图，叠加全部 7 个标的的“策略净值(归一化)”与
“买入持有基准”，并附各标的绩效汇总表（收益/年化/Sharpe/最大回撤/超额）。

用法（仓库根目录执行，venv 下）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_kelly_forest.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vnpy.trader.constant import Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_strategy import AIStrategy
from fetch_yf import YF_ASSETS, fetch_to_db

HERE = os.path.dirname(os.path.abspath(__file__))
# 干净模型：HSI 2016–2021 训练，固定权重；绝不重训，故对所有标的均无重叠泄漏
MODEL_PATH = os.path.join(HERE, "rf_model_HSI_2016.joblib")
TEST_START = "2022-01-01"
CAPITAL = 1_000_000.0  # 回测初始资金（与 set_parameters 中一致，用于重建 balance）

ASSET_ORDER = ["HSI", "CSI300", "SPY", "QQQ", "GLD", "TLT", "XOP"]
# 7 个标的的配色（策略实线 + 基准虚线同色）
COLORS = {
    "HSI":    "#d62728",  # 红
    "CSI300": "#ff7f0e",  # 橙
    "SPY":    "#1f77b4",  # 蓝
    "QQQ":    "#9467bd",  # 紫
    "GLD":    "#bcbd22",  # 黄绿（黄金）
    "TLT":    "#17becf",  # 青（国债）
    "XOP":    "#8c564b",  # 棕（油气）
}


def backtest_asset(key: str, model_path: str, test_start: str):
    """对单个标的跑回测，返回 (daily_df, 名称)。"""
    yf_sym, sym, exch, name = YF_ASSETS[key]
    vt_symbol = f"{sym}.{exch.value}"
    test_start_dt = datetime.strptime(test_start, "%Y-%m-%d")
    warmup = datetime(test_start_dt.year - 2, test_start_dt.month, test_start_dt.day)

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol, interval=Interval.DAILY,
        start=warmup, end=datetime.now(),
        rate=0.0, slippage=0.0, size=1, pricetick=0.01, capital=1_000_000,
    )
    engine.add_strategy(
        AIStrategy,
        {
            "lookback": 60, "horizon": 5, "min_train": 250,
            "retrain_interval": 20, "threshold": 0.5,
            "allow_short": False, "kelly_scale": 1.0, "max_position": 1.0,
            "stop_loss_pct": 0.05, "trailing_pct": 0.05,
            "fixed_model": True, "model_path": model_path,
            "trade_start": test_start,
        },
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    df = engine.daily_df
    # vnpy 的 daily_df 在 calculate_result 之后不含 balance 列
    # （balance 由 calculate_statistics 补上：balance = 初始资金 + 累计净盈亏）。
    # 这里手动重建，保证后续净值归一化/绩效计算可用。
    if df is not None and not df.empty and "balance" not in df.columns:
        df = df.copy()
        df["balance"] = df["net_pnl"].cumsum() + CAPITAL
    return df, name


def compute_metrics(df: pd.DataFrame, test_start_dt: datetime) -> dict:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    d = df[df.index >= test_start_dt].copy()
    if d.empty or len(d) < 2:
        return {}
    bal0, bal1 = float(d["balance"].iloc[0]), float(d["balance"].iloc[-1])
    total = bal1 / bal0 - 1.0
    days = (d.index[-1] - d.index[0]).days
    annual = (1.0 + total) ** (365.0 / days) - 1.0 if days > 0 else 0.0
    rets = d["balance"].pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    peak = d["balance"].cummax()
    dd = (d["balance"] / peak - 1.0) * 100.0
    max_dd = float(dd.min())
    hold = float(d["close_price"].iloc[-1] / d["close_price"].iloc[0] - 1.0)
    return {
        "total": total * 100.0, "annual": annual * 100.0,
        "sharpe": sharpe, "max_dd": max_dd, "hold": hold * 100.0,
    }


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"未找到干净冻结模型 {MODEL_PATH}，请确认 rf_model_HSI_2016.joblib 存在")
    test_start_dt = datetime.strptime(TEST_START, "%Y-%m-%d")

    # 1) 确保数据已拉取（统一 yfinance）
    print("===== 拉取/更新各标的日线数据 =====")
    for key in ASSET_ORDER:
        yf_sym, sym, exch, _ = YF_ASSETS[key]
        try:
            fetch_to_db(yf_sym, sym, exch)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {key}: {e}")

    # 2) 逐个跑回测，收集归一化净值与绩效
    curves = {}   # key -> (x, strat_y, bench_y)
    metrics = {}  # key -> dict
    for key in ASSET_ORDER:
        name = YF_ASSETS[key][3]
        print(f"\n===== 凯利森林算法 回测: {name} ({key}) =====")
        try:
            df, _ = backtest_asset(key, MODEL_PATH, TEST_START)
        except Exception as e:  # noqa: BLE001
            print(f"  回测失败 {key}: {e}")
            continue
        if df is None or df.empty:
            print(f"  {key} 无数据，跳过")
            continue

        x = pd.Series(pd.to_datetime(df.index))
        _mask = x >= test_start_dt
        if not _mask.any():
            print(f"  {key} 测试起点后无数据，跳过")
            continue
        i0 = int(np.argmax(_mask))
        base_b = float(df["balance"].iloc[i0])
        base_i = float(df["close_price"].iloc[i0])
        strat = (df["balance"] / base_b * 100.0).iloc[i0:]
        bench = (df["close_price"] / base_i * 100.0).iloc[i0:]
        curves[key] = (x.iloc[i0:], strat.values, bench.values)
        m = compute_metrics(df, test_start_dt)
        metrics[key] = m
        print(f"  策略收益 {m['total']:7.2f}% | 年化 {m['annual']:6.2f}% | "
              f"Sharpe {m['sharpe']:5.2f} | 最大回撤 {m['max_dd']:7.2f}% | "
              f"买入持有 {m['hold']:7.2f}% | 超额 {m['total']-m['hold']:7.2f}%")

    # 3) 单图：所有标的策略净值 + 买入持有基准
    fig = go.Figure()
    for key in ASSET_ORDER:
        if key not in curves:
            continue
        x, strat, bench = curves[key]
        color = COLORS[key]
        name = YF_ASSETS[key][3]
        fig.add_trace(go.Scatter(
            x=x, y=strat, name=f"{name}(策略)", mode="lines",
            line=dict(color=color, width=2.2),
        ))
        fig.add_trace(go.Scatter(
            x=x, y=bench, name=f"{name}(持有)", mode="lines",
            line=dict(color=color, width=1.0, dash="dot"), opacity=0.45,
        ))

    fig.update_layout(
        title=("凯利森林算法 (Kelly Forest) · 多标的样本外回测<br>"
               f"冻结随机森林(HSI 2016–2021 训练, 固定权重) + 凯利仓位 + 追踪止损<br>"
               f"测试区间 {TEST_START} 起 · 资金与指数均以测试起点归一化=100"),
        xaxis=dict(title="日期", type="date", tickformat="%Y-%m"),
        yaxis=dict(title="净值(归一化=100)", type="log"),
        height=720, width=1100,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        template="plotly_white",
    )
    fig.add_vline(x=test_start_dt, line_dash="dash", line_color="#888",
                  annotation_text="测试起点", annotation_position="top left")

    # 4) 绩效汇总表（嵌入 HTML 顶部）
    rows = []
    for key in ASSET_ORDER:
        m = metrics.get(key)
        if not m:
            continue
        name = YF_ASSETS[key][3]
        excess = m["total"] - m["hold"]
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{m['total']:.2f}%</td>"
            f"<td>{m['annual']:.2f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td style='color:#c0392b'>{m['max_dd']:.2f}%</td>"
            f"<td>{m['hold']:.2f}%</td>"
            f"<td style='color:#27ae60'>{excess:+.2f}%</td></tr>"
        )
    table_html = (
        "<div style='margin:10px 0 14px 0;font-family:sans-serif;'>"
        "<h3 style='margin:0 0 6px 0;'>凯利森林算法 · 各标的样本外绩效汇总 "
        f"(测试起点 {TEST_START})</h3>"
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse;font-size:13px;'>"
        "<thead><tr style='background:#f2f2f2;'>"
        "<th>标的</th><th>策略收益</th><th>年化收益</th><th>Sharpe</th>"
        "<th>最大回撤</th><th>买入持有</th><th>超额</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        "<p style='font-size:12px;color:#666;'>说明：模型为 HSI 2016–2021 训练的"
        "冻结随机森林（固定权重，运行期不重训），对所有标的均为样本外、无未来函数；"
        "仓位=凯利 f=2p−1，追踪止损 5%。未计手续费/滑点。</p></div>"
    )

    html_path = os.path.join(HERE, "kelly_forest_multi_chart.html")
    fig.write_html(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("<body>", "<body>" + table_html, 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n===== 多标的汇总图已生成: {html_path} =====")


if __name__ == "__main__":
    main()
