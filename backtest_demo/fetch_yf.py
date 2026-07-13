"""统一 yfinance 数据拉取：把任意 yfinance 代码写入 vnpy SQLite 数据库。

覆盖本次“凯利森林算法”多标的测试所需的所有标的：
    恒生指数 HSI、沪深300、标普500 ETF(SPY)、纳指100 ETF(QQQ)、
    黄金 ETF(GLD)、美国20+年国债 ETF(TLT)、油气资源 ETF(XOP)。

做法：
- 统一用 auto_adjust=True 取“后复权/总回报”收盘价，便于策略净值与
  “买入持有”基准在同一口径下对比（含分红再投资）。
- 列可能是 MultiIndex，压平到第一层。
- vnpy 的 save_bar_data 只接受原生 datetime（不接受 pd.Timestamp），
  所以先把含时区的索引统一转到 UTC 再 to_pydatetime()。

用法（仓库根目录执行，venv 下）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/fetch_yf.py
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy.trader.object import BarData

# yfinance 代码 -> (vnpy symbol, Exchange, 显示名)
YF_ASSETS = {
    "HSI":    ("^HSI",      "HSI",    Exchange.SEHK,  "恒生指数"),
    "CSI300": ("510300.SS", "000300", Exchange.SSE,   "沪深300"),
    "SPY":    ("SPY",       "SPY",    Exchange.SMART, "标普500"),
    "QQQ":    ("QQQ",       "QQQ",    Exchange.SMART, "纳斯达克100"),
    "GLD":    ("GLD",       "GLD",    Exchange.SMART, "黄金"),
    "TLT":    ("TLT",       "TLT",    Exchange.SMART, "美国20+国债(TLT)"),
    "XOP":    ("XOP",       "XOP",    Exchange.SMART, "油气资源(XOP)"),
}

START_DATE = "2016-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INTERVAL = Interval.DAILY

# 单根异常棒阈值：价格相对前后相邻棒都偏离超过该比例（如 50%），
# 视为 yfinance 单位/复权错乱（如把指数点位 4660 当 ETF 价 4.66 写入）。
# 真实日线极少单日波动 >50%，故该阈值不会误删正常行情。
GLITCH_PCT = 0.5


def sanitize_glitches(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, int]:
    """删除“单根孤立异常棒”：close 相对前一棒与后一棒都偏离 > GLITCH_PCT。

    典型形态：正常价 4.6 -> 异常棒 4660 -> 正常价 4.7（前后均 ~1000x/反号）。
    返回清洗后的 df 与被删除的行数。仅对内部棒（既有前驱又有后继）判定，
    避免误删序列首尾的正常大波动。
    """
    if df is None or df.empty or len(df) < 3 or "Close" not in df.columns:
        return df, 0
    c = df["Close"].to_numpy(dtype=float)
    up_from_prev = np.abs(c[1:] / c[:-1] - 1.0)
    down_to_next = np.abs(c[2:] / c[1:-1] - 1.0)
    # 对于内部棒 i (1..n-2)：与前一棒偏离 = up_from_prev[i-1]，与后一棒偏离 = down_to_next[i-1]
    bad = np.zeros(len(c), dtype=bool)
    for i in range(1, len(c) - 1):
        if up_from_prev[i - 1] > GLITCH_PCT and down_to_next[i - 1] > GLITCH_PCT:
            bad[i] = True
    n_bad = int(bad.sum())
    if n_bad:
        df = df[~bad].reset_index(drop=True)
    return df, n_bad


def fetch_to_db(yf_symbol: str, vnpy_symbol: str, exchange: Exchange,
                start: str = START_DATE, end: str = END_DATE) -> int:
    """拉取单个 yfinance 标的并 upsert 进数据库，返回写入根数。"""
    print(f"  拉取 {yf_symbol} -> {vnpy_symbol}.{exchange.value} ({start}~{end}) ...")
    df = yf.download(
        yf_symbol, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 未返回数据: {yf_symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"过滤空值后无数据: {yf_symbol}")

    # 清洗 yfinance 单根异常棒（指数点位/复权单位错乱等），避免回测虚假盈亏
    before = len(df)
    df, n_bad = sanitize_glitches(df, date_col)
    if n_bad:
        print(f"    清洗 {n_bad} 根异常棒（{before} -> {len(df)}）")

    bars: list[BarData] = []
    for _, row in df.iterrows():
        dt = pd.Timestamp(row[date_col])
        if dt.tz is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        bar = BarData(
            symbol=vnpy_symbol,
            exchange=exchange,
            datetime=dt.to_pydatetime(),
            interval=INTERVAL,
            open_price=float(row["Open"]),
            high_price=float(row["High"]),
            low_price=float(row["Low"]),
            close_price=float(row["Close"]),
            volume=float(row.get("Volume", 0) or 0),
            turnover=0.0,
            open_interest=0.0,
            gateway_name="YF",
        )
        bars.append(bar)

    ok = get_database().save_bar_data(bars)
    print(f"    写入: {'成功' if ok else '失败'} ({len(bars)} 根)")
    return len(bars)


def main() -> None:
    for key, (yf_sym, sym, exch, name) in YF_ASSETS.items():
        try:
            n = fetch_to_db(yf_sym, sym, exch)
            print(f"  [OK] {name}({key}): {n} 根")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}({key}): {e}")


if __name__ == "__main__":
    main()
