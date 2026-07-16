# coding=utf-8
# -*- coding: utf-8 -*-
"""VIX Rank 满仓/空仓 SPY 策略（纯 VIX 驱动）。

完全基于 VIX 的「长期 / 短期 Rank」（复刻 daofund 阴阳指数）决定 SPY 满仓或空仓：
  - p1 = 长期 VIX Rank（今天 VIX vs 近 long_window 日历史）
  - p2 = 短期 VIX Rank（今天 VIX vs 近 short_window 日历史）
  - p3 = 长期取反（今天 VIX vs 近 p3_window 日历史，1 - rank）
  - signal = w_p2*p2 + w_p1*p1 + w_p3*p3          （阴阳 yyi）
           或 rank_mode='p1' / 'p2' 直接使用单一 rank
  - signal < threshold  -> 满仓 SPY（用初始资金按当前价换算手数，全部买入）
  - signal >= threshold -> 空仓（全部卖出，持有现金）

无未来函数：VIX 历史在 on_init 一次性从 SQLite 全量载入；on_bar 仅用「截至当日」的
VIX 计算 rank。vnpy 回测引擎在下根 bar 开盘撮合，故决策用当日收盘 VIX、成交在下根开盘，
与 daofund 的「当日收盘决策、次日生效」一致。

依赖：numpy；信息来自 vnpy.trader.database（与 ai_strategy 同一套特征加载方式）。
"""

from datetime import datetime, date

import numpy as np

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database
from vnpy_ctastrategy import CtaTemplate, BarData


def percent_rank(values: np.ndarray, current: float) -> float:
    """复刻 daofund percentage_rank：current 在历史序列（含自身）中的经验 CDF。

    = (序列中 <= current 的个数) / len。current 越小 rank 越接近 0（VIX 低位=平静），
    current 越大 rank 越接近 1（VIX 高位=恐慌）。
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr <= current))


class VixRankStrategy(CtaTemplate):
    """VIX Rank 满仓/空仓策略。"""

    author = "demo-vix-rank"

    # ---------- 参数 ----------
    feature_symbol = "VIX"          # 信号源标的（默认 VIX）
    feature_exchange = "SMART"      # 信号源交易所
    long_window = 2520              # 长期 rank 回看窗口（~10 年交易日）
    short_window = 50              # 短期 rank 回看窗口（50 日）
    p3_window = 2000               # p3 回看窗口（长期取反）
    w_p1 = 0.15                     # 阴阳指数中 p1 权重
    w_p2 = 0.70                     # 阴阳指数中 p2 权重
    w_p3 = 0.15                     # 阴阳指数中 p3 权重
    rank_mode = "yyi"              # 'yyi' = 组合；'p1' = 仅长期；'p2' = 仅短期
    threshold = 0.70               # 信号阈值：signal < threshold 满仓，否则空仓
    use_full_capital = True        # True=满仓按初始资金换算手数（~100% 权益）；False=用 fixed_size
    fixed_size = 100               # use_full_capital=False 时的固定手数
    trade_start = ""               # 样本外起点(YYYY-MM-DD)，之前只预热不交易；留空=不限制

    # ---------- 变量 ----------
    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    signal = 0.0
    in_market = 0                  # 1=满仓, 0=空仓

    parameters = [
        "feature_symbol", "feature_exchange", "long_window", "short_window", "p3_window",
        "w_p1", "w_p2", "w_p3", "rank_mode", "threshold",
        "use_full_capital", "fixed_size", "trade_start",
    ]
    variables = ["p1", "p2", "p3", "signal", "in_market"]

    # 市价单模拟：回测引擎对限价单在「下根 bar 开盘」撮合（成交价=min(买价,下开)/max(卖价,下开)）。
    # 把下单价格设得足够极端，保证必定穿越、从而以「下根开盘价」成交（贴近真实市价单），
    # 避免在上涨市中限价单因价格不再回到原位而永远无法成交、被永远困在空仓。
    MKT_BUFFER = 1e4

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.feat_closes = np.array([], dtype=float)   # 全量 VIX 收盘（按日期升序）
        self.feat_dates = []                            # 对应日期（date 对象）
        self.trade_start_dt = None
        self.equity0 = 0.0

    def on_init(self):
        self.write_log("VIX Rank 满仓/空仓策略初始化")
        if getattr(self, "trade_start", ""):
            try:
                self.trade_start_dt = datetime.strptime(self.trade_start, "%Y-%m-%d")
            except ValueError:
                self.trade_start_dt = None
        try:
            self.equity0 = float(self.cta_engine.capital)
        except AttributeError:
            self.equity0 = 1_000_000.0
        self._load_feature_series()

    def _load_feature_series(self):
        """从 SQLite 一次性载入特征标的（VIX）全历史，按日期升序存为 feat_closes/feat_dates。"""
        try:
            fex = Exchange(self.feature_exchange) if self.feature_exchange else Exchange.SMART
        except Exception:
            fex = Exchange.SMART
        try:
            bars = get_database().load_bar_data(
                self.feature_symbol, fex, Interval.DAILY, datetime(2000, 1, 1), datetime(2100, 1, 1)
            )
        except Exception as e:
            self.write_log(f"加载特征标的 {self.feature_symbol} 失败: {e}")
            return
        rows = []
        for b in bars:
            bd = b.datetime
            if getattr(bd, "tzinfo", None) is not None:
                bd = bd.replace(tzinfo=None)
            rows.append((bd.date(), float(b.close_price)))
        rows.sort(key=lambda x: x[0])
        self.feat_dates = [r[0] for r in rows]
        self.feat_closes = np.array([r[1] for r in rows], dtype=float)
        if self.feat_closes.size:
            self.write_log(
                f"已载入特征标的 {self.feature_symbol}.{fex.value} 共 {self.feat_closes.size} 根 "
                f"({self.feat_dates[0]}~{self.feat_dates[-1]})"
            )
        else:
            self.write_log(f"特征标的 {self.feature_symbol}.{fex.value} 无数据！")

    def _idx_for(self, d: date) -> int:
        """二分查找当前日期在 feat_dates 中的位置（找不到则用最近的已存在日期）。"""
        if not self.feat_dates:
            return -1
        lo, hi = 0, len(self.feat_dates) - 1
        if d <= self.feat_dates[0]:
            return 0
        if d >= self.feat_dates[-1]:
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if self.feat_dates[mid] < d:
                lo = mid + 1
            else:
                hi = mid
        # lo 指向第一个 >= d 的位置；若严格 > d 则回退一格（用最近的历史值）
        if self.feat_dates[lo] > d and lo > 0:
            lo -= 1
        return lo

    def _compute_signal(self, idx: int) -> float:
        """用截至 idx（含）的 VIX 历史计算 p1/p2/p3/signal。"""
        if idx < 1:
            return 0.0, 0.0, 0.0, 0.0
        cur = self.feat_closes[idx]
        hist = self.feat_closes[: idx + 1]

        # 长期 rank：用 long_window 回看（不足则取全部）
        lw = min(self.long_window, len(hist))
        p1 = percent_rank(hist[-lw:], cur)
        # 短期 rank
        sw = min(self.short_window, len(hist))
        p2 = percent_rank(hist[-sw:], cur)
        # p3：长期取反
        pw = min(self.p3_window, len(hist))
        p3 = 1.0 - percent_rank(hist[-pw:], cur)

        if self.rank_mode == "p1":
            sig = p1
        elif self.rank_mode == "p2":
            sig = p2
        else:
            sig = self.w_p2 * p2 + self.w_p1 * p1 + self.w_p3 * p3
        return p1, p2, p3, sig

    def on_start(self):
        self.write_log(f"VIX Rank 策略启动（rank_mode={self.rank_mode}, threshold={self.threshold}）")

    def on_stop(self):
        self.write_log("VIX Rank 策略停止")

    def on_tick(self, tick):
        pass

    def on_bar(self, bar: BarData):
        self.cancel_all()
        bd = bar.datetime
        if getattr(bd, "tzinfo", None) is not None:
            bd = bd.replace(tzinfo=None)
        d = bd.date()

        idx = self._idx_for(d)
        if idx < 0:
            return
        self.p1, self.p2, self.p3, self.signal = self._compute_signal(idx)

        # 样本外起点之前：仅预热，不交易
        if self.trade_start_dt is not None and bd < self.trade_start_dt:
            return

        # 满仓 / 空仓 决策
        enter = self.signal < self.threshold
        if enter and not self.in_market:
            size = float(self.cta_engine.size)
            if size <= 0 or bar.close_price <= 0:
                return
            if self.use_full_capital:
                lots = int(self.equity0 / (bar.close_price * size))
            else:
                lots = int(self.fixed_size)
            if lots > 0:
                self.buy(bar.close_price * self.MKT_BUFFER, lots)
                self.in_market = 1
                self.write_log(f"满仓买入 {lots} 手 @ ~{bar.close_price:.2f}（signal={self.signal:.3f}）")
        elif not enter and self.in_market:
            if self.pos > 0:
                self.sell(bar.close_price / self.MKT_BUFFER, abs(self.pos))
                self.in_market = 0
                self.write_log(f"空仓卖出 @ ~{bar.close_price:.2f}（signal={self.signal:.3f}）")

    def on_order(self, order):
        pass

    def on_trade(self, trade):
        self.put_event()
