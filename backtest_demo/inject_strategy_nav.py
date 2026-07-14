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
    ("RF 二值（h10·每天调仓）", "rf_binary_chart.html",
     "样本外 +30.25% ｜ 超额 +26.21% ｜ Sharpe 0.466 ｜ 回撤 −21.83% ｜ 57 笔"),
    ("RF Rank 凯利版（h10·每天调仓）", "rf_rank_kelly_chart.html",
     "样本外 +8.58% ｜ 超额 +4.54% ｜ Sharpe 0.310 ｜ 回撤 −15.82% ｜ 73 笔"),
    ("RF Rank 二值版（h10·每天调仓）", "rf_rank_binary_chart.html",
     "样本外 −18.29% ｜ 超额 −22.32% ｜ Sharpe −0.136 ｜ 回撤 −47.74% ｜ 67 笔"),
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
            # 已含旧导航：先移除再重注（保证指标更新生效）
            html = re.sub(r'<div id="strat-nav".*?</div>', '', html, count=1)
            print("已含旧导航，移除后重注:", fn)
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
