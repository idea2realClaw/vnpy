#!/usr/bin/env python3
"""给 backtest_demo 下的一组策略回测 HTML 注入固定顶部导航栏（上一个/下一个策略）。

用法：
    python3 inject_strategy_nav.py

可重复运行：已含导航的 HTML 会自动跳过。导航按 STRATEGY_NAV 的顺序循环互链，
顶部状态栏同时显示当前策略名与样本外概要指标。
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# 策略顺序（循环互链）。metrics = 顶部状态栏显示的概要。
STRATEGY_NAV = [
    ("RF 二值（旧干净模型·持有5天）", "rf_binary_chart.html",
     "样本外 −21.71% ｜ 超额 −25.75% ｜ Sharpe −0.181 ｜ 回撤 −46.36% ｜ 104 笔"),
    ("RF Rank 凯利版（持有5天）", "rf_rank_kelly_chart.html",
     "样本外 +5.41% ｜ 超额 +1.38% ｜ Sharpe 0.260 ｜ 回撤 −10.13% ｜ 108 笔"),
    ("RF Rank 二值版（持有5天）", "rf_rank_binary_chart.html",
     "样本外 −10.68% ｜ 超额 −14.71% ｜ Sharpe −0.035 ｜ 回撤 −44.60% ｜ 110 笔"),
]


def inject() -> None:
    n = len(STRATEGY_NAV)
    for i, (name, fn, metrics) in enumerate(STRATEGY_NAV):
        prev = STRATEGY_NAV[(i - 1) % n]
        nxt = STRATEGY_NAV[(i + 1) % n]
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print("SKIP 不存在:", fn)
            continue
        html = open(path, encoding="utf-8").read()
        if 'id="strat-nav"' in html:
            print("已含导航，跳过:", fn)
            continue
        nav = (
            '<div id="strat-nav" style="position:fixed;top:0;left:0;right:0;z-index:99999;'
            'background:rgba(31,41,55,0.96);color:#e5e7eb;'
            'font:13px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
            'padding:9px 16px;display:flex;justify-content:space-between;align-items:center;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.35)">'
            '<a href="{prev}" style="color:#93c5fd;text-decoration:none;white-space:nowrap">'
            '← 上一个：{prev_name}</a>'
            '<span style="text-align:center;flex:1;padding:0 14px;overflow:hidden;'
            'text-overflow:ellipsis;white-space:nowrap"><b>{cur}</b> ｜ {m}</span>'
            '<a href="{next}" style="color:#93c5fd;text-decoration:none;white-space:nowrap">'
            '下一个：{next_name} →</a></div>'
        ).format(prev=prev[1], prev_name=prev[0], cur=name, m=metrics,
                 next=nxt[1], next_name=nxt[0])

        def repl(m, _nav=nav):
            return m.group(0).replace(">", ' style="margin:0;padding-top:50px">', 1) + _nav

        html = re.sub(r"<body[^>]*>", repl, html, count=1)
        open(path, "w", encoding="utf-8").write(html)
        print("已注入导航:", fn)


if __name__ == "__main__":
    inject()
