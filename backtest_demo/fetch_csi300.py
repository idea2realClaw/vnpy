"""拉取真实沪深300指数(000300.SH)日线数据并写入 SQLite，供回测使用。

数据来源：akshare 的 stock_zh_index_daily（新浪源，代码 sh000300）。
说明：akshare 的东方财富源(index_zh_a_hist)在本环境被沙箱代理拦截，
      故改用新浪源；两者均为沪深300指数真实日线。
依赖：/tmp/btvenv 已安装 akshare（pip install akshare）。
标的在 vnpy 中以 000300.SSE 表示（指数挂牌于上交所，代码 000300）。

用法（仓库根目录执行）：
    /tmp/btvenv/bin/python backtest_demo/fetch_csi300.py
"""
from datetime import datetime, timezone

import akshare as ak
import pandas as pd

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData

SYMBOL = "000300"
EXCHANGE = Exchange.SSE
INTERVAL = Interval.DAILY
START_DATE = "20220101"
# 结束日期动态取今天，重跑即自动延伸到最新交易日
END_DATE = datetime.now().strftime("%Y%m%d")


def fetch_bars() -> list[BarData]:
    """通过 akshare 拉取沪深300指数日线，转换为 BarData 列表。"""
    print("正在通过 akshare(新浪源) 拉取沪深300指数日线 ...")
    # 新浪源返回全部历史，列名为英文：date/open/high/low/close/volume
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    # 过滤到回测区间
    mask = (df["date"] >= pd.Timestamp(START_DATE)) & (df["date"] <= pd.Timestamp(END_DATE))
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("未拉取到区间内数据，请检查 akshare 源或日期范围")

    bars: list[BarData] = []
    for _, row in df.iterrows():
        # 与 gen_data.py 同一条时区链路：UTC 写入，DB 内部转上海时区存储
        d = row["date"].to_pydatetime().replace(tzinfo=timezone.utc)
        bar = BarData(
            symbol=SYMBOL,
            exchange=EXCHANGE,
            datetime=d,
            interval=INTERVAL,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row.get("volume", 0) or 0),
            turnover=0.0,
            open_interest=0.0,
            gateway_name="AKSHARE",
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

    # 打印数据库概览，确认数据已入库
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
