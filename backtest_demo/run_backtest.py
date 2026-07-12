"""对恒生指数(HSI)运行 CTA 双均线回测，并把指数走势叠加到资金曲线上。

用法（在仓库根目录执行）：
    /tmp/btvenv/bin/python backtest_demo/run_backtest.py
前置：需先用 fetch_hsi.py 把真实 HSI 日线写入数据库。
依赖：vnpy_ctastrategy（含 BacktestingEngine）/ vnpy_sqlite
    （已装在 /tmp/btvenv；数据需先由 fetch_hsi.py 写入数据库）
"""
from datetime import datetime

import pandas as pd
from vnpy.trader.constant import Exchange, Interval
from vnpy_ctastrategy.backtesting import BacktestingEngine

from double_ma_strategy import DoubleMaStrategy

VT_SYMBOL = "HSI.SEHK"            # 恒生指数（挂牌港交所，代码 HSI）
SYMBOL = "HSI"
EXCHANGE = Exchange.SEHK
INTERVAL = Interval.DAILY
# 注意：vnpy 数据库按上海时区以 naive 存储，因此 start/end 用 naive 日期，
# 结束日略放大以确保尾部 K 线被纳入。
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
        pricetick=0.01,     # 指数最小变动
        capital=1_000_000,  # 初始资金
    )

    # target_percent=1.0：做多时投入 100% 资金（满仓），策略内部按
    # capital/(close*size) 动态计算手数（HSI 约 20000 点 → ≈50 手满仓），空仓时 0%。
    engine.add_strategy(
        DoubleMaStrategy,
        {"fast_window": 10, "slow_window": 20, "target_percent": 1.0},
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

            # 确保 x 轴为真正的日期类型（vnpy daily_df 的索引是 date 对象，
            # 统一转成 pandas Timestamp，避免 plotly 当成 category 而显示不出日期）
            x_dates = pd.to_datetime(df.index)

            # 归一化：资金曲线与指数走势都除以各自起点、×100，
            # 起点统一为 100，共用一个纵坐标，涨跌幅在图上等比例可直接对比。
            base_balance = df["balance"].iloc[0]
            base_index = df["close_price"].iloc[0]
            capital_idx = df["balance"] / base_balance * 100.0      # 策略净值指数
            index_idx = df["close_price"] / base_index * 100.0      # 指数净值指数

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.08,
                specs=[[{}], [{}]],
                subplot_titles=("资金曲线 & 恒生指数走势（归一化=100）", "持仓"),
            )
            # 策略净值指数（归一化，单一纵坐标）
            fig.add_trace(
                go.Scatter(
                    x=x_dates, y=capital_idx,
                    name="策略资金净值",
                    line=dict(color="#ffc107", width=2),
                ),
                row=1, col=1,
            )
            # 沪深300指数净值（归一化，同一纵坐标，与资金曲线涨跌幅可比）
            fig.add_trace(
                go.Scatter(
                    x=x_dates, y=index_idx,
                    name="恒生指数净值",
                    line=dict(color="#1f77b4", width=1.5),
                    opacity=0.85,
                ),
                row=1, col=1,
            )
            # 持仓（底部子图）：以“持仓状态 %”展示，0% 空仓 / 100% 满仓多头，
            # 二值切换，直观呈现 0%<->100%（实际资金占比会随行情在 ~100% 附近小幅浮动）。
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
            # 显式声明 x 轴为日期类型，并格式化刻度显示
            fig.update_xaxes(
                type="date",
                tickformat="%Y-%m-%d",
                title_text="日期",
                row=1, col=1,
            )
            fig.update_xaxes(
                type="date",
                tickformat="%Y-%m-%d",
                title_text="日期",
                row=2, col=1,
            )
            fig.update_yaxes(title_text="持仓(%)", row=2, col=1)
            fig.update_layout(
                title=f"恒生指数 双均线回测 {VT_SYMBOL}", height=650
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
