"""把 vnpy 自带的 8 个 CTA 示例策略，在同一个干净切分上实跑出真实排名。

用法（仓库根目录执行，需 PYTHONPATH=仓库根）:
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_builtin_strategies.py

设置（与我们的凯利森林/CNN 回测保持一致，便于横向对比）:
    标的: 恒生指数 HSI (Exchange.SEHK, 日线)
    数据: 2016 起经 yfinance 入库（vnpy_sqlite）
    回测区间: 2020-01-01 ~ 今（含 2 年预热，确保指标充分初始化）
    样本外(OOS)窗口: 2022-01-01 ~ 今（与之前一致，无未来函数）
    资金: 1,000,000；size=10；pricetick=0.01；rate/slippage=0
    各策略用其 class 默认参数（fixed_size 等），不做调参

输出:
    backtest_demo/builtin_strategies_ranking.csv   排名表
    backtest_demo/builtin_strategies_chart.html    归一化净值曲线对比图

说明: 夏普比率与仓位规模无关（缩放仓位同时缩放收益与波动），故排名对 size 取值稳健；
      表格里的“收益%”是在 fixed_size×size=10 名义下的数值，仅供横向参考。
"""
import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from vnpy.trader.constant import Exchange, Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

# 确保 HSI 日线数据在库中
import fetch_hsi_yf
fetch_hsi_yf.main()

# (模块名, 策略类名)
BUILTINS = [
    ("atr_rsi_strategy", "AtrRsiStrategy"),
    ("boll_channel_strategy", "BollChannelStrategy"),
    ("double_ma_strategy", "DoubleMaStrategy"),
    ("dual_thrust_strategy", "DualThrustStrategy"),
    ("king_keltner_strategy", "KingKeltnerStrategy"),
    ("multi_signal_strategy", "MultiSignalStrategy"),
    ("multi_timeframe_strategy", "MultiTimeframeStrategy"),
    ("turtle_signal_strategy", "TurtleSignalStrategy"),
]

SYMBOL = "HSI"
EXCHANGE = Exchange.SEHK
VT_SYMBOL = f"{SYMBOL}.{EXCHANGE.value}"
BT_START = dt.datetime(2020, 1, 1)          # 含 2 年预热
TEST_START = dt.datetime(2022, 1, 1)        # OOS 起点
CAPITAL = 1_000_000
SIZE = 10
PRICETICK = 0.01


def oos_stats(df: pd.DataFrame, test_start_dt: dt.datetime) -> dict:
    """在 [test_start_dt, 今] 上计算无泄漏(OOS)绩效，复用与 AI 策略一致的方法。"""
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    d = df[df.index >= test_start_dt].copy()
    if d.empty or len(d) < 2:
        return None
    bal0, bal1 = float(d["balance"].iloc[0]), float(d["balance"].iloc[-1])
    total_ret = bal1 / bal0 - 1.0
    days = (d.index[-1] - d.index[0]).days
    annual = (1.0 + total_ret) ** (365.0 / days) - 1.0 if days > 0 else 0.0
    rets = d["balance"].pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    dd_col = "ddpercent" if "ddpercent" in d.columns else ("drawdown" if "drawdown" in d.columns else None)
    max_dd = float(d[dd_col].min()) if dd_col else 0.0
    if abs(max_dd) < 1:  # 比率而非百分比
        max_dd *= 100.0
    prev = d["end_pos"].shift(1).fillna(0)
    entries = int(((prev == 0) & (d["end_pos"] != 0)).sum())
    hold_ret = float(d["close_price"].iloc[-1] / d["close_price"].iloc[0] - 1.0)
    return {
        "start": d.index[0], "end": d.index[-1],
        "total_return": total_ret * 100.0, "annual_return": annual * 100.0,
        "sharpe_ratio": sharpe, "max_ddpercent": max_dd,
        "entries": entries, "hold_return": hold_ret * 100.0,
        "df": d,
    }


def run_one(cls) -> dict:
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=VT_SYMBOL, interval=Interval.DAILY,
        start=BT_START, end=dt.datetime.now(),
        rate=0.0, slippage=0.0, size=SIZE, pricetick=PRICETICK, capital=CAPITAL,
    )
    engine.add_strategy(cls, {})  # 用 class 默认参数
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    df = engine.daily_df
    return oos_stats(df, TEST_START)


def main():
    rows = []
    charts = {}
    bench = None
    for mod_name, cls_name in BUILTINS:
        try:
            mod = __import__(f"vnpy_ctastrategy.strategies.{mod_name}", fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception as e:
            print(f"[SKIP] {cls_name}: import error {e}")
            continue
        try:
            st = run_one(cls)
        except Exception as e:
            print(f"[ERR] {cls_name}: {type(e).__name__}: {e}")
            continue
        if not st:
            print(f"[WARN] {cls_name}: 无 OOS 数据")
            continue
        bench = st["hold_return"]  # 同一窗口的买入持有基准（每个策略相同）
        rows.append({
            "strategy": cls_name,
            "total_return_%": round(st["total_return"], 2),
            "annual_%": round(st["annual_return"], 2),
            "sharpe": round(st["sharpe_ratio"], 3),
            "max_dd_%": round(st["max_ddpercent"], 2),
            "entries": st["entries"],
            "hold_return_%": round(st["hold_return"], 2),
            "excess_%": round(st["total_return"] - st["hold_return"], 2),
        })
        # 归一化净值曲线（OOS 起点=100）
        eq = st["df"]["balance"] / st["df"]["balance"].iloc[0] * 100.0
        charts[cls_name] = eq
        print(f"  {cls_name:22s} 收益 {st['total_return']:7.2f}%  Sharpe {st['sharpe_ratio']:6.3f}  "
              f"最大回撤 {st['max_ddpercent']:7.2f}%  入场 {st['entries']}")

    if not rows:
        print("无可用结果")
        return

    # 排名：先按 Sharpe 降序（夏普最稳健），再按收益降序
    df = pd.DataFrame(rows).sort_values(["sharpe", "total_return_%"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    print("\n===== vnpy 内置策略 OOS 排名 (HSI, 2022-01-01~今) =====")
    print(df.to_string(index=False))
    if bench is not None:
        print(f"\n买入持有基准(B&H)同期: {bench:.2f}%")

    # 保存 CSV
    csv_path = os.path.join(HERE, "builtin_strategies_ranking.csv")
    df.drop(columns=["hold_return_%"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n排名表已保存: {csv_path}")

    # 绘图
    fig = go.Figure()
    if bench is not None and charts:
        # B&H 曲线：用第一根策略曲线的日期轴，按比例到 100
        first = next(iter(charts.values()))
        bh = first / first.iloc[0] * 100.0 * (1 + bench / 100.0)
        fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="买入持有 B&H",
                                 line=dict(color="black", width=2, dash="dash")))
    for name, eq in charts.items():
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name, line=dict(width=1.5)))
    fig.update_layout(
        title="vnpy 内置 CTA 策略 OOS 净值对比 (HSI, 2022-01-01~, 起点=100)",
        xaxis_title="日期", yaxis_title="净值 (OOS 起点=100)",
        hovermode="x unified", template="plotly_white", height=620,
    )
    html_path = os.path.join(HERE, "builtin_strategies_chart.html")
    fig.write_html(html_path)
    print(f"净值曲线图已保存: {html_path}")


if __name__ == "__main__":
    main()
