"""拉取真实恒生指数(HSI)日线数据并写入 SQLite，供回测使用。

数据来源：新浪财经港股指数接口
    https://finance.sina.com.cn/stock/hkstock/HSI/klc2_kl.js
该接口返回 JS 加密数据，需用 akshare 内置的 hk_js_decode（py_mini_racer）
解密。注意：akshare 的 stock_hk_index_daily_sina 在本环境会因上游格式
变化而 KeyError，故这里直接复刻其“抓原始 JS + 解密”的逻辑。

说明：本环境 akshare 的东方财富源被沙箱代理拦截（ProxyError），港股指数的
新浪封装函数又损坏，因此走原始新浪 URL。

依赖：/tmp/btvenv 已安装 akshare（含 py_mini_racer）。
标的在 vnpy 中以 HSI.SEHK 表示（恒生指数挂牌于港交所，代码 HSI）。

用法（仓库根目录执行）：
    /tmp/btvenv/bin/python backtest_demo/fetch_hsi.py
"""
from datetime import datetime, timezone

import pandas as pd
import requests
import py_mini_racer
from akshare.index.index_stock_hk import hk_js_decode

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData

SYMBOL = "HSI"
EXCHANGE = Exchange.SEHK
INTERVAL = Interval.DAILY
START_DATE = "20220101"
# 结束日期动态取今天，重跑即自动延伸到最新交易日
END_DATE = datetime.now().strftime("%Y%m%d")


def fetch_bars() -> list[BarData]:
    """通过新浪港股指数接口拉取 HSI 日线，解密后转换为 BarData 列表。"""
    print("正在通过新浪(港股指数)接口拉取恒生指数 HSI 日线 ...")
    url = f"https://finance.sina.com.cn/stock/hkstock/{SYMBOL}/klc2_kl.js"
    # d 参数为“起始提示”，新浪通常返回全量历史；不足时可在 Python 端按日期过滤
    r = requests.get(url, params={"d": "2010_1_01"}, timeout=30)
    if r.status_code != 200 or len(r.text) < 50:
        raise RuntimeError(f"新浪 HSI 接口返回异常: status={r.status_code}")
    raw = r.text.split("=")[1].split(";")[0].replace('"', "")
    ctx = py_mini_racer.MiniRacer()
    ctx.eval(hk_js_decode)
    records = ctx.call("d", raw)
    if not records:
        raise RuntimeError("解密后无数据，请检查 hk_js_decode 或接口变更")

    df = pd.DataFrame(records)
    # 原始 date 形如 '2013-08-20T00:00:00.000Z'（UTC）
    df["date"] = pd.to_datetime(df["date"], utc=True)
    # 过滤到回测区间
    mask = (df["date"] >= pd.Timestamp(START_DATE, tz="UTC")) & (
        df["date"] <= pd.Timestamp(END_DATE, tz="UTC")
    )
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("未拉取到区间内数据，请检查日期范围")

    bars: list[BarData] = []
    for _, row in df.iterrows():
        # 与 CSI300 同一条时区链路：UTC 写入，DB 内部转上海时区存储
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
            turnover=float(row.get("amount", 0) or 0),
            open_interest=0.0,
            gateway_name="SINA_HSI",
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
