"""AI 机器学习 CTA 策略（演示用）。

思路：用历史行情构造技术指标特征，训练一个随机森林分类器，
预测未来 N 日（horizon）指数是涨是跌；涨则做多、跌则空仓。
采用 walk-forward（滚动重训）方式，训练只用“截至当前 bar”的历史，
严格避免未来函数（look-ahead bias）。

风控：采用追踪止损——持仓盈利时止损价随持仓期间高点(低点)上移(下移)，锁定部分利润；
亏损超过初始止损线 stop_loss_pct 立即平仓。trailing_pct 控制盈利后回撤容忍度，
追踪逻辑优先级高于模型信号。

仓位（核心改动）：不再用“满仓/空仓”两档，而是按凯利公式根据模型推理出的上涨概率 p
计算目标仓位比例：target% = kelly_scale * (p - (1-p)) = kelly_scale * (2p - 1)。
- p > 0.5 做多，p 越大仓位越重；p = 0.5 空仓；p < 0.5 在 allow_short=False 时空仓，
  在 allow_short=True 时反手做空（仓位 = p-(1-p) 可为负）。
- max_position 限制单标的仓位上限；kelly_scale 缩放（1.0=满凯利，0.5=半凯利）。
- 仓位以“占总权益百分比”为准，按当前权益动态换算手数，每日随信号上调/下调/平仓。

默认多头-only（不允许做空，符合原需求）；设置 allow_short=True 可开启双向。

依赖：numpy（已装）、scikit-learn（需 pip install scikit-learn）。
"""

import datetime
import numpy as np

from vnpy.trader.constant import Direction

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


# ---------- 5 日涨幅归一化（统一函数，训练/推理共用） ----------
# 把 horizon 日涨幅 r 用「训练集涨幅的 Rank 百分位」归一化到 0~100（即 0~1）：
#   - 跌 (r < 0)              → 0
#   - r >= 0                  → (训练集中 <= r 的样本数) / (训练集样本总数) * 100
#                                也就是该涨幅在训练分布里所处的百分位百分比。
# 训练时它作为“胜率 P”的标签；回测时 CNN 直接输出这个 P（0~100%），再用凯利定律算仓位。
# 关键点：归一化的 Rank 来自「训练集」的涨幅分布，测试/推理时不接触测试数据，无未来函数。


def NormalizeReturn(train_values, r: float) -> float:
    """把单条 horizon 日涨幅 r 用训练集 train_values 的 Rank 百分位归一化到 0~100。

    对训练集「全部」涨幅（含 <0 的亏损）一起升序 Rank 排序：
      - 名次 = 训练集中 <= r 的样本数（含负值样本）
      - 归一化值 = 名次 / 训练样本总数 * 100，即 r 在训练分布里的百分位百分比
    负数涨幅不会归零，而是排在低百分位（如最差涨幅→≈0%，略亏→较小百分比）；
    涨幅越大百分位越高（最大→100%）。训练与回测统一使用本函数：
    训练时 train_values = 训练集全部 forward 涨幅；回测时模型直接输出该百分位 P。
    """
    arr = np.sort(np.asarray(train_values, dtype=float))
    rank = float(np.searchsorted(arr, r, side="right"))  # 落在 r 及以下的样本数（含负值）
    return (rank / len(arr)) * 100.0


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


def make_dataset(prices: np.ndarray, lookback: int, horizon: int, regression: bool = False):
    """构造训练集 (X, y)。遍历历史点，特征在 i 及之前，标签在 i 之后。

    regression=False（默认，分类）：标签 = 未来 horizon 日涨跌 (1=涨 / 0=跌)。
    regression=True （回归）：标签 = NormalizeReturn(训练集全部涨幅(含亏损), 未来 horizon 日涨幅) / 100，
        即「该涨幅在训练集涨幅分布中的 Rank 百分位」缩到 [0,1]，作为胜率 P
        （RF 回归器 / CNN 共用同一套标签口径，回测按凯利 f=2P-1 算仓位）。
    """
    X, y = [], []
    n = len(prices)
    if regression:
        idxs = list(range(lookback, n - horizon))
        futures = np.array(
            [prices[i + horizon] / prices[i] - 1.0 for i in idxs], dtype=float
        )
    for i in range(lookback, n - horizon):
        f = feature_at(prices, i)
        if regression:
            fut = prices[i + horizon] / prices[i] - 1.0
            lab = NormalizeReturn(futures, fut) / 100.0
        else:
            lab = label_at(prices, i, horizon)
            if lab < 0:
                continue
        X.append(f)
        y.append(lab)
    if not X:
        return np.empty((0, 0)), np.empty((0,))
    return np.array(X), np.array(y)


def cnn_image(prices: np.ndarray, i: int, lookback: int) -> np.ndarray:
    """把第 i 根 bar 之前(含)的 lookback 个收盘价，归一化后铺成 L×L 的二维“价格图”。

    - 取窗口 window = prices[i-lookback+1 : i+1]，以窗口首根价为基准做相对价
      （对绝对价格水平不变，便于跨标的迁移）。
    - 用 Hankel(滑动)方式展开成 L×L 图像：img[a, b] = 归一化价[a+b]。
      这是“纯 AI”做法——不依赖任何手工技术指标，由 CNN 自己学形态。
    - 返回 shape (L, L, 1)，可直接喂给 2D CNN。
    构建只用 i 及之前的信息，无未来函数。
    """
    window = np.asarray(prices[i - lookback + 1: i + 1], dtype=np.float32)
    base = float(window[0]) if window[0] != 0 else 1.0
    yv = window / base - 1.0
    L = lookback
    img = np.zeros((L, L), dtype=np.float32)
    for a in range(L):
        for b in range(L - a):
            img[a, b] = yv[a + b]
    return img.reshape(L, L, 1)


def make_cnn_dataset(prices: np.ndarray, lookback: int, horizon: int, regression: bool = False):
    """构造 CNN 训练集 (X, y)：每个样本是位置 i 的 L×L 价格图。

    regression=False（默认，分类）：标签 = 未来 horizon 日涨跌 (1=涨 / 0=跌)。
    regression=True （回归）：标签 = NormalizeReturn(训练集涨幅分布, 未来 horizon 日涨幅) / 100，
        即「该涨幅在训练集涨幅分布中的百分位(0~100%)」再缩到 [0,1]，作为胜率 P
        （训练集涨幅分布内 Rank：r<0→0，r>=0→百分位）。误差用 mse 训练。
    """
    X, y = [], []
    L = lookback
    n = len(prices)
    # 先收集训练集里所有样本的「未来 horizon 日涨幅」，用于回归模式的 Rank 百分位归一化
    futures = np.array(
        [prices[i + horizon] / prices[i] - 1.0 for i in range(L - 1, n - horizon)],
        dtype=float,
    )
    for i in range(L - 1, n - horizon):
        fut = prices[i + horizon] / prices[i] - 1.0
        if regression:
            # 用训练集全部涨幅的 Rank 百分位归一化（r<0→0），再 /100 缩到 [0,1] 喂给 sigmoid
            lab = NormalizeReturn(futures, fut) / 100.0
        else:
            lab = 1 if fut > 0 else 0
        X.append(cnn_image(prices, i, L))
        y.append(lab)
    if not X:
        return np.empty((0, L, L, 1)), np.empty((0,))
    return np.array(X), np.array(y)


class AIStrategy(CtaTemplate):
    """AI 择时策略（凯利百分比仓位：target% = kelly_scale*(p_up-(1-p_up))，带追踪止损）。

    两种冻结模型：
    - 分类模式 (regression=False)：模型输出涨跌概率 p_up∈[0,1]，凯利 f=2p-1。
    - 回归模式 (regression=True)：RF 回归器 / CNN 输出 P=该 5 日涨幅在训练集「全部涨幅(含亏损)」中的 Rank 百分位(0~100%)，
      再以 /100 作为胜率 p∈[0,1]，按凯利 f=2p-1 算仓位
      （亏损排低百分位→小 P→空仓/做空；百分位越高仓位越重；满百分位→满仓）。
    """

    author = "demo-ai"

    # 参数
    lookback = 60          # 特征所需最少历史根数
    horizon = 5            # 预测未来 N 日涨跌
    hold_period = 5        # 持仓持有根数：开仓后持有这么多根 bar 再重新决策；默认=horizon，与训练标签（未来 N 日涨跌）对齐；设 1 退化回每日调仓
    min_train = 250        # 至少积累这么多根才开始交易
    retrain_interval = 20  # 每多少根 bar 滚动重训一次
    threshold = 0.5        # 涨概率阈值
    allow_short = False    # 是否允许做空（默认 False）；False 时 p<=0.5 直接空仓
    kelly_scale = 1.0      # 凯利系数缩放：仓位 = kelly_scale*(p-(1-p))；1.0=满凯利，0.5=半凯利
    max_position = 1.0     # 单标的仓位上限（占总权益比例），1.0=不封顶
    use_kelly = True       # True=凯利百分比仓位；False=二值仓位(满仓100%/空仓0%)
    stop_loss_pct = 0.05   # 初始止损线：开仓后允许的最大浮亏（多头 below entry*(1-sl)，空头 above entry*(1+sl)）
    trailing_pct = 0.05    # 追踪止损幅度：盈利后止损价随高点(低点)上(下)移锁利润；设 0 退化为固定止损
    fixed_model = True     # True=加载冻结模型，不再重训（固定参数、跨标的推理）
    model_path = "rf_model.joblib"  # 固定模型文件（相对 ai_strategy.py 所在目录，或绝对路径）
    trade_start = ""       # 样本外测试起点 (YYYY-MM-DD)；留空=不限制。该日之前只预热积累特征、不交易
    model_type = "rf"      # 冻结模型类型：rf=随机森林(13维特征)；cnn=2D CNN(价格图)。加载模型时由 blob 覆盖
    regression = False     # 回归模式：CNN 输出为归一化 5 日涨幅 P∈[0,1]（而非涨跌概率）。
                           # 回测时用凯利定律 f=2P-1 把 P 换算成仓位比例。加载模型时由 blob 覆盖

    prior_adjust = False  # 先验校正：开启后把 RF 分类输出的涨概率去训练集先验偏置(P0)，并按 (1-P0) 二值化

    # 变量
    p_up = 0.0
    target_pct = 0.0       # 当前目标仓位比例（占总权益）
    model_trained = 0
    n_features_ = 0        # 冻结模型期望的特征维数
    entry_price = 0.0      # 当前持仓开仓参考价（= avg_cost）
    peak_price = 0.0       # 持仓期间跟踪的高点（多头用）
    trough_price = 0.0     # 持仓期间跟踪的低点（空头用）
    stop_price = 0.0       # 当前生效的止损价（追踪动态更新）
    equity0 = 0.0          # 初始权益（用于按百分比换算手数）
    realized_pnl = 0.0     # 累计已实现盈亏
    avg_cost = 0.0         # 当前持仓加权成本价

    parameters = [
        "lookback", "horizon", "min_train", "retrain_interval",
        "threshold", "allow_short", "kelly_scale", "max_position", "use_kelly",
        "stop_loss_pct", "trailing_pct", "fixed_model", "model_path", "trade_start",
        "regression", "hold_period", "prior_adjust",
    ]
    variables = [
        "p_up", "target_pct", "model_trained", "entry_price",
        "peak_price", "trough_price", "stop_price",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.prices = []
        self.model = None
        self.bars_since_train = 0
        self.n_features_ = 0
        self.trade_start_dt = None
        self.am = ArrayManager()
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.trough_price = 0.0
        self.stop_price = 0.0
        self.equity0 = 0.0
        self.realized_pnl = 0.0
        self.avg_cost = 0.0
        self.target_pct = 0.0
        self.bars_held = 0       # 当前持仓已持有的根数（配合 hold_period）
        self.p0 = None            # 训练集涨概率先验(P0)，由模型 blob 的 pos_rate 注入；None=未知不校正

    def _prior_correct(self, p) -> float:
        """先验校正（去训练集先验偏置）：P = P1*(1-P0) / (P1*(1-P0) + (1-P1)*P0)。
        仅当开启 prior_adjust 且已知训练集涨概率 p0 时生效；否则原样返回。
        """
        if not getattr(self, "prior_adjust", False) or not self.p0:
            return float(p)
        eps = 1e-9
        p1 = min(max(float(p), eps), 1.0 - eps)
        p0 = min(max(self.p0, eps), 1.0 - eps)
        num = p1 * (1.0 - p0)
        den = num + (1.0 - p1) * p0
        return num / den

    def get_target_pct(self, p_up: float) -> float:
        """目标仓位比例（占总权益）。

        use_kelly=False  → 二值仓位：p_up>=threshold 满仓 100%(=max_position)，否则空仓 0%；
                           即“满仓/空仓”两档，去掉凯利百分比缩放。
        use_kelly=True   → 凯利百分比仓位：f = p-(1-p) = 2p-1；kelly_scale 缩放，max_position 封顶。
        allow_short=False 时凯利仓位下限封 0（p<=0.5 即空仓）。
        """
        if not self.use_kelly:
            thr = (1.0 - self.p0) if (getattr(self, "prior_adjust", False) and self.p0) else self.threshold
            return self.max_position if p_up >= thr else 0.0
        kelly = (p_up - (1.0 - p_up)) * self.kelly_scale
        if not self.allow_short:
            kelly = max(0.0, kelly)
        return max(-self.max_position, min(self.max_position, kelly))

    def current_equity(self, price: float) -> float:
        """估算当前权益 = 初始资金 + 已实现盈亏 + 浮动盈亏（用于按百分比换算手数）。"""
        eq = self.equity0 + self.realized_pnl
        size = float(self.cta_engine.size)
        if self.pos > 0 and self.avg_cost > 0:
            eq += self.pos * size * (price - self.avg_cost)
        elif self.pos < 0 and self.avg_cost != 0:
            eq += abs(self.pos) * size * (self.avg_cost - price)
        return eq

    def get_target_lots(self, p_up: float, price: float) -> int:
        """把目标仓位比例换算成目标手数（带符号：正=多，负=空）。"""
        size = float(self.cta_engine.size)
        if size <= 0 or price <= 0:
            return 0
        eq = self.current_equity(price)
        if eq <= 0:
            return 0
        target_value = self.get_target_pct(p_up) * eq
        lots = int(target_value / (price * size))
        return lots

    def _rebalance_to(self, target_lots: int, price: float) -> None:
        """把当前持仓调整到目标手数（带符号），自动处理平多/平空/反手。"""
        delta = target_lots - self.pos
        if delta == 0:
            return
        if delta > 0:
            if self.pos < 0:
                self.buy(price, abs(self.pos))      # 先平空
                remaining = delta - abs(self.pos)
                if remaining > 0:
                    self.buy(price, remaining)      # 再开多
            else:
                self.buy(price, delta)              # 加多/开多
        else:
            neg = -delta
            if self.pos > 0:
                self.sell(price, min(neg, self.pos))  # 先平多
                remaining = neg - self.pos
                if remaining > 0:
                    self.sell(price, remaining)     # 再开空
            else:
                self.sell(price, neg)               # 加空/开空

    def on_init(self):
        self.write_log("AI 策略初始化")
        # 解析样本外测试起点（该日之前只预热、不交易）
        if getattr(self, "trade_start", ""):
            try:
                self.trade_start_dt = datetime.datetime.strptime(
                    self.trade_start, "%Y-%m-%d"
                )
                self.write_log(f"样本外测试起点: {self.trade_start}（之前仅预热）")
            except ValueError:
                self.trade_start_dt = None
                self.write_log(f"trade_start 格式错误，忽略: {self.trade_start}")
        # 记录初始权益（按百分比换算手数的基准）
        try:
            self.equity0 = float(self.cta_engine.capital)
        except AttributeError:
            self.equity0 = 1_000_000.0
        # 固定模型模式：在回测开始前一次性加载冻结模型，运行期不再重训
        if self.fixed_model:
            self._load_model()

    def _load_model(self):
        """从 model_path 加载冻结的随机森林模型（joblib）。"""
        import joblib
        import os

        path = self.model_path
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"固定模型文件不存在: {path}\n请先运行 train_model.py 生成 rf_model.joblib"
            )
        blob = joblib.load(path)
        self.model_type = blob.get("model_type", "rf")
        self.n_features_ = int(blob.get("n_features", 0))
        self.regression = bool(blob.get("regression", False))
        if self.model_type == "cnn":
            # CNN 模型：从 config+weights 重建 keras 模型（joblib 不存原始 model 对象）
            import tensorflow as tf
            mdl = tf.keras.models.model_from_json(blob["model_config"])
            mdl.set_weights(blob["model_weights"])
            self.model = mdl
            self.lookback = int(blob.get("lookback", self.lookback))
            self.write_log(
                f"已加载 CNN 固定模型: {os.path.basename(path)} "
                f"(输入 {self.lookback}x{self.lookback} 价格图, 训练源={blob.get('source')}, "
                f"回归模式={'是' if self.regression else '否'})"
            )
        else:
            self.model = blob["model"]
            self.write_log(
                f"已加载固定模型: {os.path.basename(path)} "
                f"(n_features={self.n_features_}, 训练源={blob.get('source')})"
            )
        self.p0 = blob.get("pos_rate", self.p0)   # 训练集正类率(先验 P0)，用于先验校正
        self.model_trained = 1

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

        # 持仓持有期（与训练 horizon 对齐）：开仓后持有 hold_period 根 bar 再重新决策，
        # 避免“用未来 N 日标签训练、却每天换仓”的错配。
        #   hold_period == 1：每天重新决策，pos 跨天持续持有 —— 完全等价于旧的每日调仓逻辑，
        #                    这样可以用 hold_period=1 精确还原改动前的旧绩效，做对照。
        #   hold_period  > 1：持有期内不调仓、不重新预测；到第 hold_period 根收盘平仓，
        #                    本根不再决策，下一根再重新开仓（持有期严格 = hold_period 根）。
        if self.pos != 0 and self.hold_period > 1:
            self.bars_held += 1
            if self.bars_held >= self.hold_period:
                # 持有到期：先平仓，本根不再重新决策，等下一根再开仓（逻辑最干净）
                if self.pos > 0:
                    self.sell(bar.close_price, abs(self.pos))
                else:
                    self.buy(bar.close_price, abs(self.pos))
                self._reset_position_state()
                self.bars_held = 0
                return
            else:
                return  # 持有期内不调仓、不重新预测

        # 滚动重训（walk-forward）——仅在非固定模型模式下执行
        if not self.fixed_model:
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
        else:
            # 固定模型：运行期不重训，仅记录已用训练轮数（无害）
            self.bars_since_train += 1

        if self.model is None:
            return

        if self.model_type == "cnn":
            # 纯 AI CNN：把最近 lookback 根收盘价铺成 L×L 价格图，直接推理
            if n < self.lookback:
                return
            X = cnn_image(prices, n - 1, self.lookback)        # (L, L, 1)
            X = np.expand_dims(X, axis=0)                       # (1, L, L, 1) 加 batch 维
            pred = self.model.predict(X, verbose=0)
            if self.regression:
                # 回归模式：CNN 输出单值 = 该 5 日涨幅在「训练集全部涨幅(含亏损)」中的 Rank 百分位(0~100%)，
                # 已 /100 落在 [0,1]，直接作为凯利胜率 p（亏损排低百分位→小 P；百分位越高仓位越重）。
                self.p_up = float(np.clip(np.squeeze(pred), 0.0, 1.0))
            else:
                proba = np.squeeze(pred)
                self.p_up = float(proba[1]) if len(proba) > 1 else 0.5
                self.p_up = self._prior_correct(self.p_up)  # 分类概率路径先验校正
        else:
            feat = feature_at(prices, n - 1)
            # 固定模型模式下校验特征维数一致，避免训练/推理错位
            if self.n_features_ and len(feat) != self.n_features_:
                self.write_log(
                    f"特征维数不匹配: 推理 {len(feat)} != 模型 {self.n_features_}，跳过本根"
                )
                return
            if self.regression:
                # 回归模式：RF 回归器直接输出 P = 训练集涨幅 Rank 百分位(0~1)，
                # 作为凯利胜率 p（亏损排低百分位→小 P；百分位越高仓位越重）。
                val = float(self.model.predict(feat.reshape(1, -1))[0])
                self.p_up = float(np.clip(val, 0.0, 1.0))
            else:
                proba = self.model.predict_proba(feat.reshape(1, -1))[0]
                self.p_up = float(proba[1]) if proba.shape[0] > 1 else 0.5
                self.p_up = self._prior_correct(self.p_up)  # 分类概率路径先验校正

        # 样本外测试起点之前：仅预热，不参与交易（杜绝用测试期数据做决策/训练）
        if self.trade_start_dt is not None:
            bd = bar.datetime.replace(tzinfo=None) if getattr(bar.datetime, "tzinfo", None) else bar.datetime
            if bd < self.trade_start_dt:
                return

        # 凯利百分比仓位：target% = kelly_scale*(p_up-(1-p_up)) = kelly_scale*(2p-1)
        # 每日把持仓调整到目标手数（带符号），自动上调/下调/平仓/反手。
        self.target_pct = self.get_target_pct(self.p_up)
        target_lots = self.get_target_lots(self.p_up, bar.close_price)
        self._rebalance_to(target_lots, bar.close_price)

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        """成交后更新加权成本价、已实现盈亏与止损锚点（引擎在调用前已更新 self.pos）。"""
        size = float(self.cta_engine.size)
        px = float(trade.price)
        vol = float(trade.volume)
        # 还原成交前仓位，便于更新成本/盈亏
        prev_pos = self.pos - vol if trade.direction == Direction.LONG else self.pos + vol
        if trade.direction == Direction.LONG:
            if prev_pos <= 0:
                if prev_pos < 0:                        # 平空
                    closed = min(vol, -prev_pos)
                    self.realized_pnl += (self.avg_cost - px) * closed * size
                if self.pos > 0:
                    self.avg_cost = px                  # 新开多（含先平空再开多）
                # self.pos == 0：恰好平空，avg_cost 已为 0
            else:                                       # 加多：加权平均成本
                self.avg_cost = (self.avg_cost * prev_pos + px * vol) / self.pos
            # 部分平多（prev_pos>0 且 self.pos>0）时 avg_cost 保持不变
        else:  # SHORT
            if prev_pos >= 0:
                if prev_pos > 0:                        # 平多
                    closed = min(vol, prev_pos)
                    self.realized_pnl += (px - self.avg_cost) * closed * size
                if self.pos < 0:
                    self.avg_cost = px                  # 新开空（含先平多再开空）
                # self.pos == 0：恰好平多，avg_cost 已为 0
            else:                                       # 加空：加权平均成本
                self.avg_cost = (self.avg_cost * prev_pos - px * vol) / self.pos
            # 部分平空（prev_pos<0 且 self.pos<0）时 avg_cost 保持不变

        # 更新止损锚点
        if self.pos > 0:
            self.entry_price = self.avg_cost
            if prev_pos == 0:                           # 新开多
                self.peak_price = self.avg_cost
                self.trough_price = self.avg_cost
                self.stop_price = self.avg_cost * (1.0 - self.stop_loss_pct)
        elif self.pos < 0:
            self.entry_price = self.avg_cost
            if prev_pos == 0:                           # 新开空
                self.trough_price = self.avg_cost
                self.peak_price = self.avg_cost
                self.stop_price = self.avg_cost * (1.0 + self.stop_loss_pct)
        else:
            self._reset_position_state()
        self.put_event()

    # ---------- 持仓/止损状态管理 ----------
    def _reset_position_state(self):
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.trough_price = 0.0
        self.stop_price = 0.0
        self.bars_held = 0
