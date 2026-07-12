"""双均线 CTA 策略（演示用，可回测）。

多头-only 版本：快线向上穿越慢线 -> 满仓做多(100%)；
快线向下穿越慢线 -> 清仓(0%)。全程不做空。
依赖 vnpy_ctastrategy 的 CtaTemplate。
"""

from vnpy_ctastrategy import (
    CtaTemplate,
    ArrayManager,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
)


class DoubleMaStrategy(CtaTemplate):
    """双均线交叉策略（0% 空仓 / 100% 满仓多头，禁止做空）"""

    author = "demo"

    # 参数
    fast_window = 10
    slow_window = 20
    target_percent = 1.0      # 做多时投入资金占比（1.0 = 100% 满仓）
    fixed_size = 1            # 拿不到资金量时的回退手数

    # 变量（回测/实盘时会被记录）
    fast_ma0 = 0.0
    fast_ma1 = 0.0
    slow_ma0 = 0.0
    slow_ma1 = 0.0
    target_volume = 0         # 当前信号对应的满仓手数

    parameters = ["fast_window", "slow_window", "target_percent", "fixed_size"]
    variables = ["fast_ma0", "fast_ma1", "slow_ma0", "slow_ma1", "target_volume"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.fast_ma0 = 0.0
        self.fast_ma1 = 0.0
        self.slow_ma0 = 0.0
        self.slow_ma1 = 0.0
        self.target_volume = 0
        self.am = ArrayManager()  # 1.4.x 需自行创建指标容器

    def get_target_volume(self, price: float) -> int:
        """计算 100% 满仓对应的合约手数（向下取整，至少 1 手）。

        回测时从 BacktestingEngine 取 capital / size 精确计算；
        实盘环境拿不到则回退到 fixed_size。
        """
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
        """"""
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        """"""
        self.write_log("策略启动（多头-only，0%~100% 仓位）")

    def on_stop(self):
        """"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """"""
        pass

    def on_bar(self, bar: BarData):
        """"""
        self.cancel_all()

        self.am.update_bar(bar)
        if not self.am.inited:
            return

        fast = self.am.sma(self.fast_window, array=True)
        slow = self.am.sma(self.slow_window, array=True)

        self.fast_ma0 = fast[-1]
        self.fast_ma1 = fast[-2]
        self.slow_ma0 = slow[-1]
        self.slow_ma1 = slow[-2]

        cross_over = (
            self.fast_ma0 > self.slow_ma0 and self.fast_ma1 <= self.slow_ma1
        )
        cross_below = (
            self.fast_ma0 < self.slow_ma0 and self.fast_ma1 >= self.slow_ma1
        )

        # 多头-only：空仓(0%) <-> 满仓(100%) 两态切换，禁止做空
        if cross_over:
            if self.pos == 0:
                self.target_volume = self.get_target_volume(bar.close_price)
                self.buy(bar.close_price, self.target_volume)
        elif cross_below:
            if self.pos > 0:
                self.target_volume = 0
                self.sell(bar.close_price, abs(self.pos))

    def on_order(self, order: OrderData):
        """"""
        pass

    def on_trade(self, trade: TradeData):
        """"""
        self.put_event()
