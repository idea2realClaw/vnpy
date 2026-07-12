"""AI 机器学习 CTA 策略（演示用）。

思路：用历史行情构造技术指标特征，训练一个随机森林分类器，
预测未来 N 日（horizon）指数是涨是跌；涨则做多、跌则空仓。
采用 walk-forward（滚动重训）方式，训练只用“截至当前 bar”的历史，
严格避免未来函数（look-ahead bias）。

风控：采用追踪止损——持仓盈利时止损价随持仓期间高点(低点)上移(下移)，锁定部分利润；
亏损超过初始止损线 stop_loss_pct 立即平仓。trailing_pct 控制盈利后回撤容忍度，
追踪逻辑优先级高于模型信号。

默认多头-only（不允许做空，符合原需求）；设置 allow_short=True 可开启双向。

依赖：numpy（已装）、scikit-learn（需 pip install scikit-learn）。
"""

import numpy as np

from vnpy_ctastrategy import (
    CtaTemplate,
    ArrayManager,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
)

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:  # pragma: no cover
    RandomForestClassifier = None


# ---------- 特征工程（纯 numpy，不依赖 TA-Lib，避免运行时缺库） ----------

def _sma(arr: np.ndarray, n: int) -> float:
    if len(arr) < n:
        return float("nan")
    return float(arr[-n:].mean())


def _rsi(arr: np.ndarray, n: int = 14) -> float:
    if len(arr) < n + 1:
        return float("nan")
    gains, losses = [], []
    for j in range(-n, 0):
        d = arr[j + 1] - arr[j]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = np.mean(gains), np.mean(losses)
    if al == 0.0:
        return 100.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def feature_at(prices: np.ndarray, i: int) -> np.ndarray:
    """构造第 i 根 bar 的特征向量（只用 i 及之前的信息）。"""
    c = prices[: i + 1]
    feat = []
    # 滞后日收益率
    for k in (1, 2, 3, 5, 10, 20):
        if len(c) > k:
            feat.append(c[-1] / c[-1 - k] - 1.0)
        else:
            feat.append(0.0)
    # 均线比值（短期/长期 - 1）
    for a, b in ((5, 20), (10, 60)):
        ma_a, ma_b = _sma(c, a), _sma(c, b)
        if ma_b and not np.isnan(ma_a) and not np.isnan(ma_b):
            feat.append(ma_a / ma_b - 1.0)
        else:
            feat.append(0.0)
    # 动量（20 日）
    if len(c) > 20:
        feat.append(c[-1] / c[-20] - 1.0)
    else:
        feat.append(0.0)
    # 波动率（近 20 日收益标准差）
    if len(c) > 21:
        rets = np.diff(c[-21:])
        feat.append(float(np.std(rets)))
    else:
        feat.append(0.0)
    # RSI（归一到约 [-0.5, 0.5]）
    r = _rsi(c, 14)
    feat.append((r / 100.0 - 0.5) if not np.isnan(r) else 0.0)
    # 距 60 日高/低
    win = c[-60:] if len(c) >= 60 else c
    hi, lo = float(np.max(win)), float(np.min(win))
    feat.append(c[-1] / hi - 1.0 if hi else 0.0)
    feat.append(c[-1] / lo - 1.0 if lo else 0.0)
    vec = np.array(feat, dtype=float)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def label_at(prices: np.ndarray, i: int, horizon: int) -> int:
    """第 i 根 bar 之后 horizon 日的涨跌标签（1=涨, 0=跌）。"""
    if i + horizon >= len(prices):
        return -1
    fut = prices[i + horizon] / prices[i] - 1.0
    return 1 if fut > 0 else 0


def make_dataset(prices: np.ndarray, lookback: int, horizon: int):
    """构造训练集 (X, y)。遍历历史点，特征在 i 及之前，标签在 i 之后。"""
    X, y = [], []
    for i in range(lookback, len(prices) - horizon):
        f = feature_at(prices, i)
        lab = label_at(prices, i, horizon)
        if lab < 0:
            continue
        X.append(f)
        y.append(lab)
    if not X:
        return np.empty((0, 0)), np.empty((0,))
    return np.array(X), np.array(y)


class AIStrategy(CtaTemplate):
    """随机森林 AI 择时策略（0% 空仓 / 100% 满仓多头，可开双向，带追踪止损）"""

    author = "demo-ai"

    # 参数
    lookback = 60          # 特征所需最少历史根数
    horizon = 5            # 预测未来 N 日涨跌
    min_train = 250        # 至少积累这么多根才开始交易
    retrain_interval = 20  # 每多少根 bar 滚动重训一次
    threshold = 0.5        # 涨概率阈值
    allow_short = False    # 是否允许做空（默认 False）
    target_percent = 1.0
    fixed_size = 1
    stop_loss_pct = 0.05   # 初始止损线：开仓后允许的最大浮亏（多头 below entry*(1-sl)，空头 above entry*(1+sl)）
    trailing_pct = 0.05    # 追踪止损幅度：盈利后止损价随高点(低点)上(下)移锁利润；设 0 退化为固定止损

    # 变量
    p_up = 0.0
    target_volume = 0
    model_trained = 0
    entry_price = 0.0      # 当前持仓开仓参考价
    peak_price = 0.0       # 持仓期间跟踪的高点（多头用）
    trough_price = 0.0     # 持仓期间跟踪的低点（空头用）
    stop_price = 0.0       # 当前生效的止损价（追踪动态更新）

    parameters = [
        "lookback", "horizon", "min_train", "retrain_interval",
        "threshold", "allow_short", "target_percent", "fixed_size",
        "stop_loss_pct", "trailing_pct",
    ]
    variables = [
        "p_up", "target_volume", "model_trained", "entry_price",
        "peak_price", "trough_price", "stop_price",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.prices = []
        self.model = None
        self.bars_since_train = 0
        self.am = ArrayManager()
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.trough_price = 0.0
        self.stop_price = 0.0

    def get_target_volume(self, price: float) -> int:
        try:
            capital = float(self.cta_engine.capital)
            size = float(self.cta_engine.size)
        except AttributeError:
            return max(int(self.fixed_size), 1)
        if capital <= 0 or price <= 0 or size <= 0:
            return max(int(self.fixed_size), 1)
        vol = int(capital * self.target_percent / (price * size))
        return max(vol, 1)

    def on_init(self):
        self.write_log("AI 策略初始化")

    def on_start(self):
        mode = "多空双向" if self.allow_short else "多头-only"
        self.write_log(
            f"AI 策略启动（{mode}，初始止损 {self.stop_loss_pct:.1%}，追踪 {self.trailing_pct:.1%}）"
        )

    def on_stop(self):
        self.write_log("AI 策略停止")

    def on_tick(self, tick: TickData):
        pass

    def on_bar(self, bar: BarData):
        self.cancel_all()
        self.am.update_bar(bar)
        self.prices.append(float(bar.close_price))
        prices = np.array(self.prices, dtype=float)
        n = len(prices)
        if n < self.min_train:
            return  # 积累历史，暂不交易

        # 追踪止损优先：持仓期间止损价随高低点动态上移(下移)锁利润，优先于模型信号
        if self.pos > 0 and self.entry_price > 0:
            # 更新持仓期间高点，止损价 = max(初始止损价, 峰值*(1-追踪幅度))
            self.peak_price = max(self.peak_price, bar.high_price)
            self.stop_price = max(
                self.entry_price * (1.0 - self.stop_loss_pct),
                self.peak_price * (1.0 - self.trailing_pct),
            )
            if bar.close_price <= self.stop_price:
                ep, sp = self.entry_price, self.stop_price
                self.sell(bar.close_price, abs(self.pos))    # 多头追踪止损
                self.write_log(
                    f"多头追踪止损 @ {bar.close_price:.2f}（入场 {ep:.2f}，止损价 {sp:.2f}）"
                )
                self._reset_position_state()
                return
        elif self.pos < 0 and self.entry_price > 0:
            self.trough_price = min(self.trough_price, bar.low_price)
            self.stop_price = min(
                self.entry_price * (1.0 + self.stop_loss_pct),
                self.trough_price * (1.0 + self.trailing_pct),
            )
            if bar.close_price >= self.stop_price:
                ep, sp = self.entry_price, self.stop_price
                self.buy(bar.close_price, abs(self.pos))     # 空头追踪止损
                self.write_log(
                    f"空头追踪止损 @ {bar.close_price:.2f}（入场 {ep:.2f}，止损价 {sp:.2f}）"
                )
                self._reset_position_state()
                return

        # 滚动重训（walk-forward）
        self.bars_since_train += 1
        need_retrain = (self.model is None) or (self.bars_since_train >= self.retrain_interval)
        if need_retrain and RandomForestClassifier is not None:
            X, y = make_dataset(prices, self.lookback, self.horizon)
            if len(X) >= 30:
                self.model = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
                )
                self.model.fit(X, y)
                self.bars_since_train = 0
                self.model_trained = 1
                self.write_log(f"模型重训完成，样本数={len(X)}")

        if self.model is None:
            return

        feat = feature_at(prices, n - 1)
        proba = self.model.predict_proba(feat.reshape(1, -1))[0]
        self.p_up = float(proba[1]) if proba.shape[0] > 1 else 0.5
        signal_up = self.p_up >= self.threshold

        vol = self.get_target_volume(bar.close_price)
        if signal_up:
            if self.pos < 0:
                self.buy(bar.close_price, abs(self.pos) + vol)  # 平空 + 开多
                self._enter_long(bar.close_price)
            elif self.pos == 0:
                self.buy(bar.close_price, vol)                  # 开多
                self._enter_long(bar.close_price)
            # self.pos > 0：已持有多头，不重复下单
        else:
            if self.pos > 0:
                self.sell(bar.close_price, abs(self.pos))       # 平多
                self._reset_position_state()
            elif self.pos == 0 and self.allow_short:
                self.sell(bar.close_price, vol)                 # 开空
                self._enter_short(bar.close_price)
            # self.pos < 0：已持有空头，不重复下单

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.put_event()

    # ---------- 持仓/止损状态管理 ----------
    def _enter_long(self, price: float):
        self.entry_price = price
        self.peak_price = price
        self.trough_price = price
        self.stop_price = price * (1.0 - self.stop_loss_pct)

    def _enter_short(self, price: float):
        self.entry_price = price
        self.trough_price = price
        self.peak_price = price
        self.stop_price = price * (1.0 + self.stop_loss_pct)

    def _reset_position_state(self):
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.trough_price = 0.0
        self.stop_price = 0.0
