"""抓取 VTS 组合策略轮动所需的 ETF / 波动率指数日线并写入 SQLite。

数据来源：Yahoo Finance (yfinance)。
参考 https://www.volatilitytradingstrategies.com/portfolio 的子策略轮动，
需要以下标的（vnpy 中以 .SMART 表示，与 SPY.SMART / VIX.SMART 同风格）：

  QLD   2x Nasdaq-100 ETF      -> Defensive Rotation 的“低波动=加仓”档
  XLU   公用事业 ETF           -> Defensive Rotation 的“中高波动=防御”档
  IYR   房地产 ETF             -> Strategic Tail Risk 的“中波动”档
  VIXM  VIX 中期期货 ETF        -> Strategic Tail Risk 的“极端波动=做多波动率”档
  VXV   CBOE 3 月波动率指数(^VIX3M) -> 与 VIX 组成期限结构 VIX/VXV，做 Barometer 分量
  VVIX  CBOE VVIX 指数(^VVIX)   -> vol of vol，做 Barometer 分量（VTS 13 指标之 #6/#13）

SPY.SMART 与 VIX.SMART 已存在，本脚本只补上述 6 个。
复用 fetch_vix.py 的 BarData 写入范式（走 vnpy on_conflict_replace upsert）。

用法（仓库根目录执行）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/fetch_vts_etfs.py
"""
from datetime import datetime

import pandas as pd
import yfinance as yf

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData

INTERVAL = Interval.DAILY
# 延伸到 2006：让 VIX 长期百分位(p1)的 10 年回看窗口有真实历史上下文
# （否则 2016-2017 因样本不足导致 p1 偏低、误判为极端低波动而过度空仓）。
START_DATE = "2006-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# yfinance 代码 -> (vnpy symbol, exchange)
ASSETS = {
    "^VIX": ("VIX", Exchange.SMART),   # 已存在，重跑仅延伸
    "SPY": ("SPY", Exchange.SMART),    # 延伸到 2006（原仅 2016+），使 Barometer 全历史锚点可校验
    "QLD": ("QLD", Exchange.SMART),
    "XLU": ("XLU", Exchange.SMART),
    "IYR": ("IYR", Exchange.SMART),
    "VIXM": ("VIXM", Exchange.SMART),
    # ^VXV 已在 yfinance 下架；^VIX3M（CBOE 3 月波动率指数）是其等价物，
    # VIX/VXV 期限结构即 VIX/VIX3M。仍存为 VXV.SMART 以保持 Barometer 代码一致。
    "^VIX3M": ("VXV", Exchange.SMART),
    # ^VVIX = CBOE VVIX（VIX 的波动率，vol of vol）。VTS Barometer 13 指标中
    # 含 VVIX 水平(#6) 与 VVIX/VIX 比(#13)；本地此前用代理，现补真实数据以提升吻合度。
    "^VVIX": ("VVIX", Exchange.SMART),
}


def fetch_bars(yf_sym: str, symbol: str, exchange: Exchange) -> list[BarData]:
    print(f"拉取 {yf_sym} ({symbol}.{exchange.value}) {START_DATE} ~ {END_DATE} ...")
    df = yf.download(yf_sym, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        print(f"  !! {yf_sym} 无数据，跳过")
        return []
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    if df.empty:
        print(f"  !! {yf_sym} 过滤后无数据，跳过")
        return []

    bars: list[BarData] = []
    for _, row in df.iterrows():
        dt = pd.Timestamp(row["Date"])
        dt = dt.tz_convert("UTC") if dt.tz is not None else dt.tz_localize("UTC")
        bars.append(BarData(
            symbol=symbol, exchange=exchange, datetime=dt.to_pydatetime(),
            interval=INTERVAL,
            open_price=float(row["Open"]), high_price=float(row["High"]),
            low_price=float(row["Low"]), close_price=float(row["Close"]),
            volume=float(row.get("Volume", 0) or 0),
            turnover=0.0, open_interest=0.0, gateway_name="YF_VTS",
        ))
    return bars


def main() -> None:
    db = get_database()
    for yf_sym, (symbol, exchange) in ASSETS.items():
        bars = fetch_bars(yf_sym, symbol, exchange)
        if not bars:
            continue
        ok = db.save_bar_data(bars)
        print(f"  写入 {symbol}.{exchange.value}: {'成功' if ok else '失败'} "
              f"({len(bars)} 根, {bars[0].datetime.date()}~{bars[-1].datetime.date()})")


if __name__ == "__main__":
    main()
