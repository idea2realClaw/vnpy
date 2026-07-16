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
# 全部使用 h10 冻结模型（HSI 2016–2021 训练，无重训），每天调仓(hold=1)，样本外测试。
STRATEGY_NAV = [
    # —— 恒生指数 HSI（OOS 2022-01-03~，买入持有 +4.03%）——
    ("HSI · RF 二值（h10·分类）", "rf_binary_chart.html",
     "样本外 +30.25% ｜ 超额 +26.21% ｜ Sharpe 0.466 ｜ 回撤 −21.83% ｜ 57 笔"),
    ("HSI · RF Rank 凯利版（h10·回归）", "rf_rank_kelly_chart.html",
     "样本外 +8.58% ｜ 超额 +4.54% ｜ Sharpe 0.310 ｜ 回撤 −15.82% ｜ 73 笔"),
    ("HSI · RF Rank 二值版（h10·回归）", "rf_rank_binary_chart.html",
     "样本外 −18.29% ｜ 超额 −22.32% ｜ Sharpe −0.136 ｜ 回撤 −47.74% ｜ 67 笔"),
    # —— 沪深300 CSI300（OOS 2022-01-04~，买入持有 +8.90%）——
    ("CSI300 · RF 二值（h10·分类）", "csi300_rf_binary_chart.html",
     "样本外 +1.09% ｜ 超额 −7.81% ｜ Sharpe 0.091 ｜ 回撤 −30.88% ｜ 52 笔"),
    ("CSI300 · RF Rank 凯利版（h10·回归）", "csi300_rf_rank_kelly_chart.html",
     "样本外 +0.52% ｜ 超额 −8.38% ｜ Sharpe 0.049 ｜ 回撤 −8.88% ｜ 68 笔"),
    ("CSI300 · RF Rank 二值版（h10·回归）", "csi300_rf_rank_binary_chart.html",
     "样本外 +9.40% ｜ 超额 +0.50% ｜ Sharpe 0.219 ｜ 回撤 −24.84% ｜ 67 笔"),
    # —— 标普500 SPY（OOS 2022-01-03~，买入持有 +67.24%）——
    ("SPY · RF 二值（h10·分类）", "spy_rf_binary_chart.html",
     "样本外 +49.66% ｜ 超额 −17.58% ｜ Sharpe 0.645 ｜ 回撤 −21.81% ｜ 87 笔"),
    ("SPY · RF Rank 凯利版（h10·回归）", "spy_rf_rank_kelly_chart.html",
     "样本外 +2.63% ｜ 超额 −64.61% ｜ Sharpe 0.167 ｜ 回撤 −5.21% ｜ 64 笔"),
    ("SPY · RF Rank 二值版（h10·回归）", "spy_rf_rank_binary_chart.html",
     "样本外 +35.53% ｜ 超额 −31.70% ｜ Sharpe 0.550 ｜ 回撤 −16.27% ｜ 61 笔"),
    # —— 标普500 SPY（VIX 特征训练·h10，OOS 2022-01-03~，买入持有约 +58%）——
    # 用 VIX 形态预测 SPY 涨跌：分类二值首次跑赢 SPY 买入持有基准；回归Rank 在 VIX 特征下 P 趋中、空仓。
    ("SPY·VIX · RF 二值（h10·分类）", "spy_vix_rf_binary_chart.html",
     "样本外 +64.51% ｜ 超额 +6.50% ｜ Sharpe 0.720 ｜ 回撤 −22.33% ｜ 7 笔（跑赢基准）"),
    ("SPY·VIX · RF Rank 凯利版（h10·回归）", "spy_vix_rf_rank_kelly_chart.html",
     "样本外 0.00% ｜ 0 笔交易（回归Rank P 趋中，空仓）"),
    ("SPY·VIX · RF Rank 二值版（h10·回归）", "spy_vix_rf_rank_binary_chart.html",
     "样本外 0.00% ｜ 0 笔交易（回归Rank P 趋中，空仓）"),
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
