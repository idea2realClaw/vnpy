"""训练一次随机森林并冻结保存为固定模型（不再 walk-forward 重训）。

用法（在仓库根目录执行）：
    # 全量训练（默认，向后兼容）
    /tmp/btvenv/bin/python backtest_demo/train_model.py --source HSI
    /tmp/btvenv/bin/python backtest_demo/train_model.py --source CSI300

    # train/test 切分：只用 2022~2024 训练，2025 起做样本外测试
    /tmp/btvenv/bin/python backtest_demo/train_model.py \
        --source HSI --train-end 2024-12-31 \
        --out backtest_demo/rf_model_HSI_2024.joblib

输出：rf_model.joblib（或 --out 指定的路径）
    存 {model, n_features, lookback, horizon, source, train_end, trained_at}

说明：
    之后任何标的都用这个“固定模型”做推理（run_any_backtest.py），
    回测过程中不再重新训练。特征全为无量纲量（收益率 / 均线比值 /
    RSI / 距高低），所以一个在 HSI 上训好的模型可以迁移到 CSI300
    等其它标的——这正是“固定参数 + 任意标的推理”的核心前提。
依赖：scikit-learn（已装在 /tmp/btvenv）
"""
import argparse
import datetime
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_strategy import make_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def get_bars(source: str):
    """返回训练用的 BarData 列表（策略只用其中的收盘价）。"""
    if source.upper() == "HSI":
        import fetch_hsi
        return fetch_hsi.fetch_bars()
    if source.upper() == "CSI300":
        import fetch_csi300
        return fetch_csi300.fetch_bars()
    raise ValueError(f"未知训练源: {source}（目前支持 HSI / CSI300）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="HSI", help="训练源标的: HSI / CSI300")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument(
        "--train-end", default=None,
        help="训练数据截止日 (YYYY-MM-DD)。指定后模型只用该日及之前的数据训练，"
             "之后的数据留作样本外测试，杜绝未来函数泄漏。",
    )
    ap.add_argument(
        "--out", default=None,
        help="模型输出路径。默认 rf_model.joblib；指定 --train-end 且未给 --out 时"
             "自动命名为 rf_model_<source>_<train-end>.joblib。",
    )
    args = ap.parse_args()

    bars = get_bars(args.source)
    if args.train_end:
        te = datetime.datetime.strptime(args.train_end, "%Y-%m-%d").date()
        bars = [b for b in bars if b.datetime.date() <= te]
        print(f"训练数据已截断到 {args.train_end}（仅用此日及之前，之后留作样本外）")

    prices = np.array([float(b.close_price) for b in bars], dtype=float)
    print(
        f"训练源 {args.source}: {len(prices)} 根日线, "
        f"区间 {bars[0].datetime.date()} ~ {bars[-1].datetime.date()}"
    )

    X, y = make_dataset(prices, args.lookback, args.horizon)
    if len(X) < 30:
        raise RuntimeError("样本不足，无法训练（至少需要 30 条）")
    print(f"训练样本 X={X.shape}, 正类(涨)占比={y.mean():.2%}")

    model = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    print("随机森林训练完成（固定参数，不再重训）")

    # 决定输出路径
    if args.out:
        out = args.out
    elif args.train_end:
        out = os.path.join(HERE, f"rf_model_{args.source}_{args.train_end}.joblib")
    else:
        out = os.path.join(HERE, "rf_model.joblib")
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)

    joblib.dump(
        {
            "model": model,
            "n_features": int(X.shape[1]),
            "lookback": args.lookback,
            "horizon": args.horizon,
            "source": args.source,
            "train_end": args.train_end,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        out,
    )
    print(f"固定模型已保存: {out}")
    if args.train_end:
        print(
            "样本外测试命令（例如）：\n"
            f"  /tmp/btvenv/bin/python backtest_demo/run_any_backtest.py "
            f"--target {args.source} --model {os.path.basename(out)} "
            f"--test-start 2025-01-01"
        )


if __name__ == "__main__":
    main()
