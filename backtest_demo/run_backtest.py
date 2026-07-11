"""对沪深300指数运行 CTA 双均线回测，并把指数走势叠加到资金曲线上。

用法（在仓库根目录执行）：
    /tmp/btvenv/bin/python backtest_demo/run_backtest.py
前置：需先用 fetch_csi300.py 把真实指数日线写入数据库。
依赖：vnpy_ctastrategy（含 BacktestingEngine）/ vnpy_sqlite
    （已装在 /tmp/btvenv；数据需先由 fetch_csi300.py 写入数据库）
"""
from datetime import datetime

from vnpy.trader.constant import Exchange, Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

from double_ma_strategy import DoubleMaStrategy

VT_SYMBOL = "000300.SSE"          # 沪深300指数（挂牌上交所，代码 000300）
SYMBOL = "000300"
EXCHANGE = Exchange.SSE
INTERVAL = Interval.DAILY
# 注意：vnpy 数据库按上海时区以 naive 存储，因此 start/end 用 naive 日期，
# 结束日略放大以确保尾部 K 线被纳入。
START = datetime(2022, 1, 1)
END = datetime(2024, 1, 10)


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
        pricetick=0.01,     # 指数最小变动
        capital=1_000_000,  # 初始资金
    )

    # fixed_size 放大到约 285，使 1 手持仓名义约 ≈ 满仓（≈100万/3500点），
    # 这样资金曲线能真实反映“择时策略 vs 买入持有指数”的差异。
    engine.add_strategy(
        DoubleMaStrategy,
        {"fast_window": 10, "slow_window": 20, "fixed_size": 285},
    )

    engine.load_data()
    engine.run_backtesting()

    engine.calculate_result()
    stats = engine.calculate_statistics()   # 打印绩效指标，并补充 balance 列

    df = engine.daily_df
    if df is not None and not df.empty:
        csv_path = "backtest_demo/backtest_daily_result.csv"
        df.to_csv(csv_path)
        print(f"\n每日资金曲线已导出: {csv_path}")

        # 生成收益曲线 HTML 图表（若有 plotly）
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.08,
                specs=[[{"secondary_y": True}], [{}]],
                subplot_titles=("资金曲线 & 沪深300指数走势", "持仓"),
            )
            # 资金曲线（左轴）
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["balance"],
                    name="策略资金曲线",
                    line=dict(color="#ffc107", width=2),
                ),
                row=1, col=1, secondary_y=False,
            )
            # 沪深300指数收盘（右轴，作为参考/基准叠加）
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["close_price"],
                    name="沪深300指数(收)",
                    line=dict(color="#1f77b4", width=1.5),
                    opacity=0.85,
                ),
                row=1, col=1, secondary_y=True,
            )
            # 持仓（底部子图）
            if "end_pos" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df["end_pos"],
                        name="持仓",
                        line=dict(color="#2ca02c", width=1.2),
                        fill="tozeroy",
                    ),
                    row=2, col=1,
                )
            fig.update_yaxes(title_text="资金(元)", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="指数点位", row=1, col=1, secondary_y=True)
            fig.update_layout(
                title=f"沪深300指数 双均线回测 {VT_SYMBOL}", height=650
            )
            html_path = "backtest_demo/backtest_chart.html"
            fig.write_html(html_path)
            print(f"收益曲线图表(含指数叠加): {html_path}")
        except Exception as e:  # noqa: BLE001
            print(f"跳过图表生成: {e}")

    print("\n===== 绩效统计 =====")
    for k, v in (stats or {}).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
