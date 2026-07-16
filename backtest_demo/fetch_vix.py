"""拉取 VIX（恐慌指数）日线全历史并写入 SQLite，供“用 VIX 特征训练 SPY”使用。

数据来源：Yahoo Finance (yfinance)，代码 ^VIX。
- VIX 是 CBOE 波动率指数，与 SPY 高度负相关，可作为 SPY 涨跌的辅助特征。
- 写入走 vnpy 的 on_conflict_replace（按 symbol/exchange/interval/datetime 主键 upsert）。

标的在 vnpy 中以 VIX.SMART 表示（与 SPY.SMART 同风格，便于策略 on_init 加载）。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/fetch_vix.py
"""
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData

SYMBOL = "VIX"
EXCHANGE = Exchange.SMART
INTERVAL = Interval.DAILY
START_DATE = "2016-01-01"
# 结束日期动态取今天，重跑即自动延伸到最新交易日
END_DATE = datetime.now().strftime("%Y-%m-%d")


def fetch_bars() -> list[BarData]:
    """通过 yfinance 拉取 VIX 日线全历史，转换为 BarData 列表。"""
    print(f"正在通过 yfinance 拉取 VIX 日线 {START_DATE} ~ {END_DATE} ...")
    df = yf.download(
        "^VIX", start=START_DATE, end=END_DATE,
        auto_adjust=False, progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance 未返回 VIX 数据，请检查网络或代码 ^VIX")
    # 新版本 yfinance 列是 MultiIndex (字段, ticker)，这里压平到第一层
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("过滤空值后无数据")

    bars: list[BarData] = []
    for _, row in df.iterrows():
        dt = pd.Timestamp(row["Date"])
        if dt.tz is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        # vnpy 的 save_bar_data 只接受原生 Python datetime（不接受 pd.Timestamp）
        dt_py = dt.to_pydatetime()
        bar = BarData(
            symbol=SYMBOL,
            exchange=EXCHANGE,
            datetime=dt_py,
            interval=INTERVAL,
            open_price=float(row["Open"]),
            high_price=float(row["High"]),
            low_price=float(row["Low"]),
            close_price=float(row["Close"]),
            volume=float(row.get("Volume", 0) or 0),
            turnover=0.0,
            open_interest=0.0,
            gateway_name="YF_VIX",
        )
        bars.append(bar)
    return bars


def main() -> None:
    db = get_database()
    bars = fetch_bars()
    print(
        f"拉取 {len(bars)} 根日线，"
        f"区间 {bars[0].datetime.date()} ~ {bars[-1].datetime.date()}"
    )

    ok = db.save_bar_data(bars)
    print(f"写入数据库: {'成功' if ok else '失败'}")

    for o in db.get_bar_overview():
        if o.symbol == SYMBOL:
            start = o.start.date() if o.start else None
            end = o.end.date() if o.end else None
            print(
                f"  {o.symbol}.{o.exchange.value} [{o.interval.value}] "
                f"共{o.count}根 {start}~{end}"
            )


if __name__ == "__main__":
    main()
