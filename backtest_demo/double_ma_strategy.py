"""双均线 CTA 策略（演示用，可回测）。

快线向上穿越慢线 -> 做多；快线向下穿越慢线 -> 做空。
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
    """双均线交叉策略"""

    author = "demo"

    # 参数
    fast_window = 10
    slow_window = 20
    fixed_size = 1

    # 变量（回测/实盘时会被记录）
    fast_ma0 = 0.0
    fast_ma1 = 0.0
    slow_ma0 = 0.0
    slow_ma1 = 0.0

    parameters = ["fast_window", "slow_window", "fixed_size"]
    variables = ["fast_ma0", "fast_ma1", "slow_ma0", "slow_ma1"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.fast_ma0 = 0.0
        self.fast_ma1 = 0.0
        self.slow_ma0 = 0.0
        self.slow_ma1 = 0.0
        self.am = ArrayManager()  # 1.4.x 需自行创建指标容器

    def on_init(self):
        """"""
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        """"""
        self.write_log("策略启动")

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

        if cross_over:
            if self.pos == 0:
                self.buy(bar.close_price, self.fixed_size)
            elif self.pos < 0:
                self.cover(bar.close_price, abs(self.pos))
                self.buy(bar.close_price, self.fixed_size)
        elif cross_below:
            if self.pos == 0:
                self.short(bar.close_price, self.fixed_size)
            elif self.pos > 0:
                self.sell(bar.close_price, abs(self.pos))
                self.short(bar.close_price, self.fixed_size)

    def on_order(self, order: OrderData):
        """"""
        pass

    def on_trade(self, trade: TradeData):
        """"""
        self.put_event()
