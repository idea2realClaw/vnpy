"""用“固定模型”对任意标的做回测推理（不再重新训练）。

用法（在仓库根目录执行）：
    # 预设：沪深300 / 恒生指数
    /tmp/btvenv/bin/python backtest_demo/run_any_backtest.py --target CSI300
    /tmp/btvenv/bin/python backtest_demo/run_any_backtest.py --target HSI

    # 任意 A 股指数（新浪源），例如上证50 / 创业板指
    /tmp/btvenv/bin/python backtest_demo/run_any_backtest.py \
        --ak-code sh000016 --symbol 000016 --exchange SSE --name 上证50
    /tmp/btvenv/bin/python backtest_demo/run_any_backtest.py \
        --ak-code sz399006 --symbol 399006 --exchange SZSE --name 创业板指

前置：
    1) 先用 train_model.py 生成冻结模型 backtest_demo/rf_model.joblib
    2) 数据会自动抓取并写入 vnpy SQLite（A股用 akshare 新浪源，HSI 用新浪原始解码）
说明：
    模型固定（fixed_model=True），回测期间不再 walk-forward 重训；
    特征全为无量纲量，故一个在 HSI 上训好的模型可迁移到 CSI300 等其它标的。
依赖：vnpy_ctastrategy / vnpy_sqlite / scikit-learn / akshare / plotly
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData
from vnpy_ctastrategy.backtesting import BacktestingEngine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_strategy import AIStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "rf_model.joblib")

PRESETS = {
    "CSI300": {"symbol": "000300", "exchange": Exchange.SSE, "source": "csi300", "name": "沪深300"},
    "HSI": {"symbol": "HSI", "exchange": Exchange.SEHK, "source": "hsi_yf", "name": "恒生指数"},
}


def _exchange_from_str(s: str) -> Exchange:
    return {"SSE": Exchange.SSE, "SZSE": Exchange.SZSE, "SEHK": Exchange.SEHK}[s.upper()]


def ensure_data(cfg: dict) -> None:
    """抓取并写入数据库，确保回测引擎能 load_data。"""
    src = cfg["source"]
    if src == "csi300":
        import fetch_csi300
        fetch_csi300.main()
    elif src == "hsi":
        import fetch_hsi
        fetch_hsi.main()
    elif src == "hsi_yf":
        import fetch_hsi_yf
        fetch_hsi_yf.main()
    elif src == "akshare_index":
        import akshare as ak
        ak_code = cfg["ak_code"]
        df = ak.stock_zh_index_daily(symbol=ak_code)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        bars = []
        for _, row in df.iterrows():
            d = row["date"].to_pydatetime().replace(tzinfo=timezone.utc)
            bars.append(
                BarData(
                    symbol=cfg["symbol"], exchange=cfg["exchange"],
                    datetime=d, interval=Interval.DAILY,
                    open_price=float(row["open"]), high_price=float(row["high"]),
                    low_price=float(row["low"]), close_price=float(row["close"]),
                    volume=float(row.get("volume", 0) or 0),
                    turnover=0.0, open_interest=0.0, gateway_name="AKSHARE",
                )
            )
        ok = get_database().save_bar_data(bars)
        print(f"写入数据库: {'成功' if ok else '失败'} ({len(bars)} 根)")
    else:
        raise ValueError(f"未知数据源: {src}")


def build_cfg(args) -> dict:
    if args.target and args.target.upper() in PRESETS:
        return dict(PRESETS[args.target.upper()])
    if args.ak_code:
        exch = _exchange_from_str(args.exchange) if args.exchange else (
            Exchange.SSE if args.ak_code.startswith("sh") else Exchange.SZSE
        )
        sym = args.symbol or (args.ak_code[2:] if args.ak_code.startswith(("sh", "sz")) else args.ak_code)
        return {
            "symbol": sym, "exchange": exch, "source": "akshare_index",
            "ak_code": args.ak_code, "name": args.name or sym,
        }
    raise SystemExit("请指定 --target CSI300/HSI 或 --ak-code <新浪指数代码>")


def compute_window_stats(df: pd.DataFrame, test_start_dt: datetime):
    """在测试区间 [test_start_dt, 今] 上计算无泄漏绩效（样本外）。"""
    # daily_df 的 index 可能是 datetime.date 对象，统一转成 Timestamp 便于比较
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
    if abs(max_dd) < 1:  # 可能是比率而非百分比
        max_dd *= 100.0
    prev = d["end_pos"].shift(1).fillna(0)
    entries = int(((prev == 0) & (d["end_pos"] != 0)).sum())
    hold_ret = float(d["close_price"].iloc[-1] / d["close_price"].iloc[0] - 1.0)
    return {
        "start": d.index[0], "end": d.index[-1],
        "total_return": total_ret * 100.0, "annual_return": annual * 100.0,
        "sharpe_ratio": sharpe, "max_ddpercent": max_dd,
        "entries": entries, "hold_return": hold_ret * 100.0,
    }


def run_backtest(cfg: dict, model_path: str = MODEL_PATH, test_start: str = None) -> None:
    vt_symbol = f"{cfg['symbol']}.{cfg['exchange'].value}"
    name = cfg["name"]
    test_start_dt = None
    if test_start:
        test_start_dt = datetime.strptime(test_start, "%Y-%m-%d")
        # 预热：测试起点前 2 年数据仅用于构造首批特征，不计入交易，杜绝未来函数
        warmup_start = datetime(test_start_dt.year - 2, test_start_dt.month, test_start_dt.day)
        bt_start = warmup_start
        print(f"\n===== 固定模型回测（train/test 切分）：{name} ({vt_symbol}) =====")
        print(f"测试区间 = {test_start} ~ 今；预热数据 {warmup_start.date()} 起（仅构造特征，不交易）")
    else:
        bt_start = datetime(2022, 1, 1)
        print(f"\n===== 固定模型回测：{name} ({vt_symbol}) =====")

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=vt_symbol, interval=Interval.DAILY,
        start=bt_start, end=datetime.now(),
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
            "trade_start": test_start or "",
        },
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    df = engine.daily_df
    if df is None or df.empty:
        print("无回测数据，结束")
        return

    csv_path = os.path.join(HERE, f"frozen_{cfg['symbol']}_daily_result.csv")
    df.to_csv(csv_path)
    print(f"每日资金曲线已导出: {csv_path}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        x_dates = pd.to_datetime(df.index)
        # 归一化基准：指定样本外测试起点时，资金与指数都在该日归一到 100；
        # 不指定时沿用全区间首根。预热期资金恒定=1.0，故测试起点基准即初始资金。
        if test_start_dt is not None:
            _mask = x_dates >= test_start_dt
            if _mask.any():
                _i0 = int(np.argmax(_mask))
                base_balance = float(df["balance"].iloc[_i0])
                base_index = float(df["close_price"].iloc[_i0])
            else:
                base_balance = float(df["balance"].iloc[0])
                base_index = float(df["close_price"].iloc[0])
        else:
            base_balance = float(df["balance"].iloc[0])
            base_index = float(df["close_price"].iloc[0])
        capital_idx = df["balance"] / base_balance * 100.0
        index_idx = df["close_price"] / base_index * 100.0

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3], vertical_spacing=0.08,
            specs=[[{}], [{}]],
            subplot_titles=(
                f"{name}（固定模型，{'样本外 ' + test_start + ' 起以测试起点归一化=100' if test_start else '归一化=100'}）",
                "",
            ),
        )
        fig.add_trace(
            go.Scatter(x=x_dates, y=capital_idx, name="策略资金净值",
                       line=dict(color="#ffc107", width=2)), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_dates, y=index_idx, name=f"{name}净值(买入持有)",
                       line=dict(color="#1f77b4", width=1.5), opacity=0.85), row=1, col=1,
        )
        if "end_pos" in df.columns:
            size = 1.0  # 指数回测 size=1；仓位占比 = 手数*收盘价 / 权益
            exposure = df["end_pos"] * df["close_price"] * size / df["balance"] * 100.0
            exposure = exposure.clip(lower=0)
            fig.add_trace(
                go.Scatter(x=x_dates, y=exposure, name="仓位(%)",
                           line=dict(color="#2ca02c", width=1.2), fill="tozeroy"),
                row=2, col=1,
            )
        fig.update_yaxes(title_text="净值(归一化=100)", row=1, col=1)
        fig.update_xaxes(type="date", tickformat="%Y-%m-%d", row=1, col=1)
        fig.update_xaxes(type="date", tickformat="%Y-%m-%d", row=2, col=1)
        fig.update_yaxes(title_text="仓位(%)", row=2, col=1)

        last_date = x_dates[-1].strftime("%Y-%m-%d")
        fig.add_annotation(xref="x domain", yref="y domain", x=1, y=0,
                           xanchor="right", yanchor="bottom",
                           text=f"最新交易日: {last_date}",
                           showarrow=False, font=dict(size=11, color="#555"),
                           bgcolor="rgba(255,255,255,0.7)")
        fig.add_annotation(xref="x domain", yref="y2 domain", x=0, y=1,
                           xanchor="left", yanchor="top",
                           text="持仓", showarrow=False,
                           font=dict(size=13, color="#2ca02c"))
        fig.update_layout(title=f"{name} AI 固定模型回测 {vt_symbol}", height=650)
        if test_start_dt is not None:
            fig.add_vline(
                x=test_start_dt, line_dash="dash", line_color="#d62728",
                annotation_text="测试起点", annotation_position="top left",
            )
        html_path = os.path.join(HERE, f"frozen_{cfg['symbol']}_chart.html")
        fig.write_html(html_path)

        # 最近一周持仓表格，插入到 HTML 最上方
        import re
        lw = df.tail(5).sort_index()
        rows_html = []
        for dt, r in lw.iterrows():
            dstr = pd.to_datetime(dt).strftime("%Y-%m-%d")
            pos = int(r.get("end_pos", 0))
            balance = float(r["balance"])
            close = float(r["close_price"])
            size = 1.0  # 指数回测 size=1
            pct = (pos * close * size / balance * 100.0) if balance else 0.0
            pct = max(pct, 0.0)
            if pos > 0:
                plabel = "多头"
            elif pos < 0:
                plabel = "空头"
            else:
                plabel = "空仓"
            rows_html.append(
                f"<tr><td>{dstr}</td><td>{plabel}</td>"
                f"<td>{pct:.1f}%</td><td>¥{balance:,.2f}</td></tr>"
            )
        table_html = (
            "<div style='margin:10px 0 18px 0;font-family:sans-serif;'>"
            f"<h3 style='margin:0 0 6px 0;'>{name} 最近一周持仓（固定模型{('·样本外 ' + test_start) if test_start else ''}）</h3>"
            "<table border='1' cellpadding='6' cellspacing='0' "
            "style='border-collapse:collapse;font-size:13px;'>"
            "<thead><tr style='background:#f2f2f2;'>"
            "<th>日期</th><th>持仓</th><th>仓位(%)</th><th>总资产</th>"
            "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
        )
        with open(html_path, "r", encoding="utf-8") as _f:
            _c = _f.read()
        _c = re.sub(r"<body[^>]*>", lambda m: m.group(0) + table_html, _c, count=1)
        with open(html_path, "w", encoding="utf-8") as _f:
            _f.write(_c)
        print(f"收益曲线图表(含{name}叠加): {html_path}")
    except Exception as e:  # noqa: BLE001
        print(f"跳过图表生成: {e}")

    # 同时给出“买入持有”基准，便于对比固定模型是否跑赢
    hold_ret = (df["close_price"].iloc[-1] / df["close_price"].iloc[0] - 1) * 100
    print("\n===== 绩效统计（全区间，含预热） =====")
    for k, v in (stats or {}).items():
        print(f"{k}: {v}")
    print(f"\n[{name}] 买入持有基准收益(全区间): {hold_ret:.2f}%")

    if test_start_dt is not None:
        w = compute_window_stats(df, test_start_dt)
        if w:
            print(f"\n===== 样本外测试区间绩效（{w['start'].date()} ~ {w['end'].date()}，无未来函数） =====")
            print(f"策略总收益:   {w['total_return']:8.2f}%")
            print(f"年化收益:     {w['annual_return']:8.2f}%")
            print(f"Sharpe:       {w['sharpe_ratio']:8.3f}")
            print(f"最大回撤:     {w['max_ddpercent']:8.2f}%")
            print(f"入场次数:     {w['entries']}")
            print(f"买入持有基准: {w['hold_return']:8.2f}%")
            print(f"超额收益:     {w['total_return'] - w['hold_return']:8.2f}%")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="CSI300", help="预设标的: CSI300 / HSI")
    ap.add_argument("--ak-code", default=None, help="自定义指数新浪代码, 如 sh000016")
    ap.add_argument("--symbol", default=None, help="vnpy 标的代码, 如 000016")
    ap.add_argument("--exchange", default=None, choices=["SSE", "SZSE", "SEHK"])
    ap.add_argument("--name", default=None, help="显示名称")
    ap.add_argument("--model", default=None, help="固定模型路径（默认 rf_model.joblib）")
    ap.add_argument(
        "--test-start", default=None,
        help="样本外测试起点 (YYYY-MM-DD)。指定后仅用该日及之后做推理/交易，"
             "之前的数据作预热（模型已冻结，绝不训练）。对应 train_model.py 的 --train-end。",
    )
    args = ap.parse_args()

    model_path = MODEL_PATH
    if args.model:
        model_path = args.model if os.path.isabs(args.model) else os.path.join(HERE, args.model)
    if not os.path.exists(model_path):
        raise SystemExit(f"未找到固定模型 {model_path}，请先运行 train_model.py")

    cfg = build_cfg(args)
    ensure_data(cfg)
    run_backtest(cfg, model_path=model_path, test_start=args.test_start)


if __name__ == "__main__":
    main()
