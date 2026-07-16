"""TA-Lib 集成冒烟测试：验证 fork 内 ArrayManager 的 talib 指标可正常计算。

运行:
    PYTHONPATH=D:/Github/vnpy python tests/test_talib_arraymanager.py
"""
import math
import sys
import numpy as np

# 加载仓库内的 fork（而非 pip 安装的官方包）
import vnpy  # noqa: F401
from vnpy.trader.utility import ArrayManager


class FakeBar:
    def __init__(self, o, h, l, c, v):
        self.open_price = float(o)
        self.high_price = float(h)
        self.low_price = float(l)
        self.close_price = float(c)
        self.volume = float(v)
        self.turnover = float(v) * float(c)
        self.open_interest = 0.0


def build_bars(n=300):
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 1, n)
    close = 100.0 + np.cumsum(rets)
    high = close + np.abs(rng.normal(0, 1, n))
    low = close - np.abs(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100, 1000, n).astype(float)
    return [FakeBar(o, h, l, c, v) for o, h, l, c, v in zip(open_, high, low, close, volume)]


def check_one(name, am, fn):
    """返回 (ok, info)"""
    try:
        res = fn(am)
    except Exception as e:  # noqa: BLE001
        return False, f"EXCEPTION {type(e).__name__}: {e}"

    if isinstance(res, tuple):
        parts = []
        ok = True
        for i, x in enumerate(res):
            if isinstance(x, np.ndarray):
                if x.shape != (am.size,):
                    ok = False
                    parts.append(f"shape{x.shape}!=({am.size},)")
                else:
                    parts.append("arr ok" if np.any(np.isfinite(x)) else "arr ALLNAN")
                    if not np.any(np.isfinite(x)):
                        ok = False
            else:
                if not math.isfinite(x):
                    ok = False
                    parts.append(f"val={x}!")
                else:
                    parts.append(f"{x:.4f}")
        return ok, " | ".join(parts)

    if isinstance(res, np.ndarray):
        if res.shape != (am.size,):
            return False, f"shape{res.shape}!=({am.size},)"
        if not np.any(np.isfinite(res)):
            return False, "ALLNAN"
        return True, f"arr last={res[-1]:.4f}"

    if isinstance(res, (int, float)):
        if not math.isfinite(res):
            return False, f"val={res}!"
        return True, f"{res:.4f}"

    return False, f"UNKNOWN type {type(res)}"


def main():
    N = 300
    am = ArrayManager(size=N)
    for bar in build_bars(N):
        am.update_bar(bar)

    assert am.inited, "ArrayManager 未初始化"
    assert am.count == N, f"count={am.count}"
    print(f"ArrayManager: size={am.size} count={am.count} inited={am.inited}\n")

    cases = [
        ("sma(30)", lambda a: a.sma(30)),
        ("ema(30)", lambda a: a.ema(30)),
        ("kama(30)", lambda a: a.kama(30)),
        ("rsi(14)", lambda a: a.rsi(14)),
        ("macd(12,26,9)", lambda a: a.macd(12, 26, 9)),
        ("adx(14)", lambda a: a.adx(14)),
        ("adxr(14)", lambda a: a.adxr(14)),
        ("dx(14)", lambda a: a.dx(14)),
        ("atr(14)", lambda a: a.atr(14)),
        ("natr(14)", lambda a: a.natr(14)),
        ("cci(14)", lambda a: a.cci(14)),
        ("boll(20,2)", lambda a: a.boll(20, 2)),
        ("donchian(20)", lambda a: a.donchian(20)),
        ("sar(0.02,0.2)", lambda a: a.sar(0.02, 0.2)),
        ("stoch(5,3,0,3,0)", lambda a: a.stoch(5, 3, 0, 3, 0)),
        ("aroon(20)", lambda a: a.aroon(20)),
        ("aroonosc(20)", lambda a: a.aroonosc(20)),
        ("willr(14)", lambda a: a.willr(14)),
        ("mfi(14)", lambda a: a.mfi(14)),
        ("obv()", lambda a: a.obv()),
        ("mom(10)", lambda a: a.mom(10)),
        ("roc(10)", lambda a: a.roc(10)),
        ("trix(30)", lambda a: a.trix(30)),
        ("adosc(3,10)", lambda a: a.adosc(3, 10)),
        ("minus_dm(14)", lambda a: a.minus_dm(14)),
        ("plus_dm(14)", lambda a: a.plus_dm(14)),
        ("minus_di(14)", lambda a: a.minus_di(14)),
        ("plus_di(14)", lambda a: a.plus_di(14)),
    ]

    passed = 0
    failed = 0
    print(f"{'INDICATOR':<18} RESULT")
    print("-" * 60)
    for name, fn in cases:
        ok, info = check_one(name, am, fn)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{name:<18} [{status}] {info}")

    print("-" * 60)
    print(f"TOTAL: {passed} passed, {failed} failed, {len(cases)} cases")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
