"""生成一段合成的日线历史数据并写入 SQLite 数据库，供回测引擎读取。

数据纯属随机游走合成，仅用于演示回测流程，不具备任何真实市场含义。
"""

import random
from datetime import datetime, timedelta, timezone

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData

# 合成标的：螺纹钢期货 rb888.SHFE（仅演示用，数据为随机生成）
SYMBOL = "rb888"
EXCHANGE = Exchange.SHFE
INTERVAL = Interval.DAILY
START = datetime(2022, 1, 1, tzinfo=timezone.utc)
END = datetime(2023, 12, 31, tzinfo=timezone.utc)


def generate_bars() -> list[BarData]:
    """用几何随机游走生成每个交易日的日线。"""
    bars: list[BarData] = []
    price = 3500.0
    d = START
    rng = random.Random(20240101)  # 固定种子，结果可复现

    while d <= END:
        # 跳过周末（仅生成工作日 K 线）
        if d.weekday() < 5:
            open_price = price
            change = rng.gauss(0.0, 35.0)
            close_price = max(1.0, open_price + change)
            high_price = max(open_price, close_price) + abs(rng.gauss(0.0, 18.0))
            low_price = min(open_price, close_price) - abs(rng.gauss(0.0, 18.0))
            volume = rng.randint(120000, 600000)

            bar = BarData(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                datetime=d,
                interval=INTERVAL,
                open_price=round(open_price, 2),
                high_price=round(high_price, 2),
                low_price=round(low_price, 2),
                close_price=round(close_price, 2),
                volume=float(volume),
                open_interest=0.0,
                gateway_name="DEMO",
            )
            bars.append(bar)
            price = close_price

        d += timedelta(days=1)

    return bars


def main() -> None:
    db = get_database()
    bars = generate_bars()
    print(f"生成 {len(bars)} 根日线，时间区间 {bars[0].datetime.date()} ~ {bars[-1].datetime.date()}")

    ok = db.save_bar_data(bars)
    print(f"写入数据库: {'成功' if ok else '失败'}")

    # 打印数据库概览，确认数据已入库
    overview = db.get_bar_overview()
    print("数据库 Bar 概览:")
    for o in overview:
        start = o.start.date() if o.start else None
        end = o.end.date() if o.end else None
        print(f"  {o.symbol}.{o.exchange.value} [{o.interval.value}] 共{o.count}根 {start}~{end}")


if __name__ == "__main__":
    main()
