"""对恒生指数(HSI)运行 AI 机器学习 CTA 回测，并把指数走势叠加到资金曲线上。

用法（在仓库根目录执行）：
    /tmp/btvenv/bin/python backtest_demo/run_ai_backtest.py
前置：需先用 fetch_hsi.py 把真实 HSI 日线写入数据库。
依赖：vnpy_ctastrategy（含 BacktestingEngine）/ vnpy_sqlite / scikit-learn
    （已装在 /tmp/btvenv；数据需先由 fetch_hsi.py 写入数据库）
"""
from datetime import datetime

import pandas as pd
from vnpy.trader.constant import Exchange, Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

from ai_strategy import AIStrategy

VT_SYMBOL = "HSI.SEHK"            # 恒生指数（挂牌港交所，代码 HSI）
SYMBOL = "HSI"
EXCHANGE = Exchange.SEHK
INTERVAL = Interval.DAILY
START = datetime(2022, 1, 1)
END = datetime.now()                       # 结束日取今天，回测延伸到最新交易日


def main() -> None:
    engine = BacktestingEngine()

    engine.set_parameters(
        vt_symbol=VT_SYMBOL,
        interval=INTERVAL,
        start=START,
        end=END,
        rate=0.0,           # 手续费（按金额比例，演示设为 0）
        slippage=0.0,       # 滑点
        size=1,             # 指数：1 点 = 1 元
        pricetick=0.01,      # 指数最小变动
        capital=1_000_000,  # 初始资金
    )

    # AI 策略参数：用随机森林预测未来 5 日涨跌，涨则满仓做多、跌则空仓。
    engine.add_strategy(
        AIStrategy,
        {
            "lookback": 60, "horizon": 5, "min_train": 250,
            "retrain_interval": 20, "threshold": 0.5,
            "allow_short": False, "target_percent": 1.0,
            "stop_loss_pct": 0.05, "trailing_pct": 0.05,
        },
    )

    engine.load_data()
    engine.run_backtesting()

    engine.calculate_result()
    stats = engine.calculate_statistics()   # 打印绩效指标，并补充 balance 列

    df = engine.daily_df
    if df is not None and not df.empty:
        csv_path = "backtest_demo/ai_backtest_daily_result.csv"
        df.to_csv(csv_path)
        print(f"\n每日资金曲线已导出: {csv_path}")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            x_dates = pd.to_datetime(df.index)

            base_balance = df["balance"].iloc[0]
            base_index = df["close_price"].iloc[0]
            capital_idx = df["balance"] / base_balance * 100.0
            index_idx = df["close_price"] / base_index * 100.0

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.08,
                specs=[[{}], [{}]],
                subplot_titles=("资金曲线 & 恒生指数（AI策略，归一化=100）", ""),
            )
            fig.add_trace(
                go.Scatter(
                    x=x_dates, y=capital_idx,
                    name="策略资金净值",
                    line=dict(color="#ffc107", width=2),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_dates, y=index_idx,
                    name="恒生指数净值",
                    line=dict(color="#1f77b4", width=1.5),
                    opacity=0.85,
                ),
                row=1, col=1,
            )
            if "end_pos" in df.columns:
                pos_state = (df["end_pos"] > 0).astype(int) * 100
                fig.add_trace(
                    go.Scatter(
                        x=x_dates, y=pos_state,
                        name="持仓(%)",
                        line=dict(color="#2ca02c", width=1.2),
                        fill="tozeroy",
                    ),
                    row=2, col=1,
                )
            fig.update_yaxes(title_text="净值(归一化=100)", row=1, col=1)
            fig.update_xaxes(type="date", tickformat="%Y-%m-%d", row=1, col=1)
            fig.update_xaxes(type="date", tickformat="%Y-%m-%d", row=2, col=1)
            fig.update_yaxes(title_text="持仓(%)", row=2, col=1)

            # 第一张图：最新交易日日期标注在右下角
            last_date = x_dates[-1].strftime("%Y-%m-%d")
            fig.add_annotation(
                xref="x domain", yref="y domain", x=1, y=0,
                xanchor="right", yanchor="bottom",
                text=f"最新交易日: {last_date}",
                showarrow=False, font=dict(size=11, color="#555"),
                bgcolor="rgba(255,255,255,0.7)",
            )
            # 第二张图：持仓标题放在左上角
            fig.add_annotation(
                xref="x domain", yref="y2 domain", x=0, y=1,
                xanchor="left", yanchor="top",
                text="持仓", showarrow=False,
                font=dict(size=13, color="#2ca02c"),
            )
            fig.update_layout(
                title=f"恒生指数 AI 机器学习回测 {VT_SYMBOL}", height=650
            )
            html_path = "backtest_demo/ai_backtest_chart.html"
            fig.write_html(html_path)
            print(f"收益曲线图表(含指数叠加): {html_path}")

            # 最近一周持仓表格，插入到 HTML 最上方
            try:
                import re
                lw = df.tail(5).sort_index()
                rows_html = []
                for dt, r in lw.iterrows():
                    dstr = pd.to_datetime(dt).strftime("%Y-%m-%d")
                    pos = int(r.get("end_pos", 0))
                    plabel = "多头" if pos > 0 else ("空头" if pos < 0 else "空仓")
                    rows_html.append(
                        f"<tr><td>{dstr}</td><td>{plabel}</td>"
                        f"<td>{abs(pos)}</td><td>¥{float(r['balance']):,.2f}</td></tr>"
                    )
                table_html = (
                    "<div style='margin:10px 0 18px 0;font-family:sans-serif;'>"
                    "<h3 style='margin:0 0 6px 0;'>最近一周持仓</h3>"
                    "<table border='1' cellpadding='6' cellspacing='0' "
                    "style='border-collapse:collapse;font-size:13px;'>"
                    "<thead><tr style='background:#f2f2f2;'>"
                    "<th>日期</th><th>持仓</th><th>仓位(手)</th><th>总资产</th>"
                    "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
                )
                with open(html_path, "r", encoding="utf-8") as _f:
                    _c = _f.read()
                _c = re.sub(r"<body[^>]*>", lambda m: m.group(0) + table_html, _c, count=1)
                with open(html_path, "w", encoding="utf-8") as _f:
                    _f.write(_c)
                print("最近一周持仓表格已插入 HTML 顶部")
            except Exception as e:  # noqa: BLE001
                print(f"跳过表格插入: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"跳过图表生成: {e}")

    print("\n===== 绩效统计 =====")
    for k, v in (stats or {}).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
