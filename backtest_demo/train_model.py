"""训练并冻结保存为固定模型（不再 walk-forward 重训）。

用法（在仓库根目录执行）：
    # 随机森林（默认，13 维手工特征）
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/train_model.py --source HSI

    # 纯 AI 2D CNN（最近 lookback 根收盘价铺成 L×L 价格图，无手工特征）
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/train_model.py \
        --source HSI --model-type cnn --train-end 2021-12-31

    # train/test 切分：只用 2021 年底之前训练，2022 起做样本外测试
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/train_model.py \
        --source HSI --model-type cnn --train-end 2021-12-31 \
        --out backtest_demo/cnn_model_HSI_2021-12-31.joblib

输出：<rf|cnn>_model.joblib（或 --out 指定的路径）
    rf : 存 {model, n_features, lookback, horizon, source, train_end, trained_at}
    cnn: 存 {model_type, model_config, model_weights, n_features, lookback,
             horizon, source, train_end, trained_at}
          （不存原始 keras 对象，加载时由 config+weights 重建）

说明：
    之后任何标的都用这个“固定模型”做推理（run_any_backtest.py），
    回测过程中不再重新训练。
    - rf 特征为无量纲量（收益率 / 均线比值 / RSI / 距高低），可跨标的迁移；
    - cnn 用“相对价”归一化的价格图（对绝对价格水平不变），同样可跨标的迁移。
依赖：scikit-learn（rf）；tensorflow（cnn，需 pip install tensorflow-cpu）
"""
import argparse
import datetime
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_strategy import make_dataset, make_cnn_dataset, NormalizeReturn

HERE = os.path.dirname(os.path.abspath(__file__))

YF_START = "2016-01-01"


def resolve_yf_ticker(src: str) -> str:
    """把训练/特征源名映射成 yfinance ticker；未知则原样当作 ticker。"""
    m = {"HSI": "^HSI", "VIX": "^VIX", "SPY": "SPY"}
    return m.get(src.upper() if src else src, src)


def fetch_yf_prices(ticker: str, start: str, end: str = None):
    """通过 yfinance 拉取日线收盘价，返回 (dates: list[date], prices: np.ndarray)。"""
    print(f"  yfinance 拉取 {ticker} ({start} ~ {end or '今'}) ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 未返回 {ticker} 数据，请检查代码或网络")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    dates = [d.date() for d in df["Date"]]
    prices = np.array(df["Close"].astype(float).values, dtype=float)
    return dates, prices


def build_cnn_model(L: int, regression: bool = False):
    """纯 AI 2D CNN：输入 L×L 价格图。

    regression=False（分类，默认）：输出未来 horizon 日涨跌的 2 类概率（softmax）。
    regression=True （回归）：输出单值 P = NormalizeReturn(5日涨幅) ∈ [0,1]（sigmoid），
        训练用 mse 损失，回测时按凯利 f=2P-1 换算仓位。

    结构（纯 AI、无手工技术指标，由 CNN 自己从价格图学形态）：
      Conv2D(32,3)+BN → MaxPool → Conv2D(64,3)+BN → MaxPool
      → Conv2D(64,3)+BN → GlobalAveragePool → Dense(64)+BN → Dropout(0.4) → 输出层
    训练用 Adam(lr=1e-3) + ReduceLROnPlateau，早停：分类看 val_accuracy，回归看 val_loss。
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPool2D, GlobalAveragePooling2D, Flatten,
        Dense, Dropout, BatchNormalization,
    )

    m = Sequential([
        Input((L, L, 1)),
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D(2),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation="sigmoid") if regression else Dense(2, activation="softmax"),
    ])
    if regression:
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
    else:
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return m


def get_bars(source: str):
    """返回训练用的 BarData 列表（策略只用其中的收盘价）。"""
    if source.upper() == "HSI":
        # yfinance 全历史（2016 起），覆盖 sina 旧子集，支持 2016+ 训练
        import fetch_hsi_yf
        return fetch_hsi_yf.fetch_bars()
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
        "--model-type", default="rf", choices=["rf", "cnn"],
        help="模型类型：rf=随机森林(13维手工特征, 默认); "
             "cnn=纯AI 2D CNN(最近 lookback 根收盘价铺成价格图, 无手工特征)。",
    )
    ap.add_argument(
        "--regression", action="store_true",
        help="回归模式（CNN / RF 均适用）：标签=NormalizeReturn(5日涨幅)∈[0,1] 作为胜率 P，"
             "RF 用 RandomForestRegressor、CNN 用单值 sigmoid；回测按凯利 f=2P-1 算仓位。"
             "默认关闭(分类涨跌)。",
    )
    ap.add_argument(
        "--epochs", type=int, default=20,
        help="仅 CNN 训练用：训练轮数（默认 20）。",
    )
    ap.add_argument(
        "--out", default=None,
        help="模型输出路径。默认 rf_model.joblib(或 cnn_model.joblib)；"
             "指定 --train-end 且未给 --out 时自动命名为 <类型>_model_<source>_<train-end>.joblib。",
    )
    ap.add_argument(
        "--feature-source", default=None,
        help="特征源标的（yfinance ticker 别名，如 VIX 表示 ^VIX）。指定后进入「跨标的」模式："
             "特征来自该标的序列，标签来自 --source（如 SPY）序列，二者按日期对齐。"
             "不指定则特征=标签=--source（单序列，向后兼容）。",
    )
    args = ap.parse_args()

    if args.feature_source:
        # ---------- 跨标的模式：特征=feature-source 序列，标签=source 序列 ----------
        feat_ticker = resolve_yf_ticker(args.feature_source)
        label_ticker = resolve_yf_ticker(args.source)
        f_dates, f_prices = fetch_yf_prices(feat_ticker, YF_START, None)
        l_dates, l_prices = fetch_yf_prices(label_ticker, YF_START, None)
        f_map = dict(zip(f_dates, f_prices))
        l_map = dict(zip(l_dates, l_prices))
        common = sorted(set(f_dates) & set(l_dates))
        feature_prices = np.array([f_map[d] for d in common], dtype=float)
        label_prices = np.array([l_map[d] for d in common], dtype=float)
        print(
            f"特征源 {args.feature_source}({feat_ticker}): {len(f_prices)} 根; "
            f"标签源 {args.source}({label_ticker}): {len(l_prices)} 根; "
            f"按日期对齐后 {len(common)} 根"
        )
        if args.train_end:
            te = datetime.datetime.strptime(args.train_end, "%Y-%m-%d").date()
            mask = np.array([d <= te for d in common], dtype=bool)
            feature_prices = feature_prices[mask]
            label_prices = label_prices[mask]
            common = [d for d in common if d <= te]
            print(f"训练数据已截断到 {args.train_end}（仅用此日及之前，之后留作样本外）")
        print(
            f"对齐训练区间 {common[0]} ~ {common[-1]}, "
            f"特征(VIX类)={len(feature_prices)} 根, 标签(SPY类)={len(label_prices)} 根"
        )
        if args.model_type == "cnn":
            X, y = make_cnn_dataset(
                feature_prices, args.lookback, args.horizon,
                regression=args.regression, label_prices=label_prices,
            )
        else:
            X, y = make_dataset(
                feature_prices, args.lookback, args.horizon,
                regression=args.regression, label_prices=label_prices,
            )
    else:
        # ---------- 单序列模式（向后兼容 HSI/CSI300） ----------
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

        if args.model_type == "cnn":
            X, y = make_cnn_dataset(prices, args.lookback, args.horizon, regression=args.regression)
        else:
            X, y = make_dataset(prices, args.lookback, args.horizon, regression=args.regression)
    if len(X) < 30:
        raise RuntimeError("样本不足，无法训练（至少需要 30 条）")
    if args.regression:
        print(f"训练样本 X={X.shape}, 回归标签=训练集涨幅 Rank 百分位/100 "
              f"均值={y.mean():.4f} 中位数={np.median(y):.4f} "
              f"(即训练集内涨幅分位：0=跌, 1=最大涨幅)")
        print(f"（标签范围 {y.min():.4f} ~ {y.max():.4f}，说明 Rank 覆盖整个训练分布）")
    else:
        print(f"训练样本 X={X.shape}, 正类(涨)占比={y.mean():.2%}")

    trained_at = datetime.datetime.now().isoformat(timespec="seconds")
    if args.model_type == "cnn":
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, TerminateOnNaN,
        )
        model = build_cnn_model(args.lookback, regression=args.regression)
        print(f"CNN 训练开始（epochs={args.epochs}, 输入 {args.lookback}x{args.lookback} 价格图, "
              f"{'回归(归一化5日涨幅)' if args.regression else '分类(涨跌)'}）...")
        model.fit(
            X, y,
            epochs=args.epochs,
            batch_size=32,
            validation_split=0.15,
            verbose=2,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss" if args.regression else "val_accuracy",
                    mode="min" if args.regression else "max",
                    patience=10, restore_best_weights=True,
                    min_delta=0.005,
                ),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=5, min_lr=1e-5, verbose=1),
                TerminateOnNaN(),
            ],
        )
        print("CNN 训练完成（固定参数，不再重训）")
        meta = {
            "model_type": "cnn",
            "regression": bool(args.regression),
            "normalization": "rank_percentile_train",  # 标签=训练集涨幅 Rank 百分位/100
            "model_config": model.to_json(),
            "model_weights": model.get_weights(),
            "n_features": int(args.lookback),
            "lookback": args.lookback,
            "horizon": args.horizon,
            "source": args.source,
            "train_end": args.train_end,
            "trained_at": trained_at,
        }
        prefix = "cnn_model"
    else:
        if args.regression:
            model = RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=10,
                random_state=42, n_jobs=-1,
            )
            model.fit(X, y)
            print("随机森林(回归/Rank 胜率)训练完成（固定参数，不再重训）")
            meta = {
                "model_type": "rf",
                "regression": True,
                "normalization": "rank_percentile_train",
                "model": model,
                "n_features": int(X.shape[1]),
                "lookback": args.lookback,
                "horizon": args.horizon,
                "source": args.source,
                "train_end": args.train_end,
                "trained_at": trained_at,
            }
            prefix = "rf_ret_model"
        else:
            model = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            model.fit(X, y)
            print("随机森林训练完成（固定参数，不再重训）")
            meta = {
                "model_type": "rf",
                "model": model,
                "n_features": int(X.shape[1]),
                "lookback": args.lookback,
                "horizon": args.horizon,
                "source": args.source,
                "train_end": args.train_end,
                "trained_at": trained_at,
                "pos_rate": float(y.mean()),   # 训练集正类率(先验 P0)，供推理期先验校正使用
            }
            prefix = "rf_model"

    # 决定输出路径
    if args.out:
        out = args.out
    elif args.train_end:
        if args.feature_source:
            out = os.path.join(HERE, f"{prefix}_{args.source}_feat{args.feature_source}_{args.train_end}.joblib")
        else:
            out = os.path.join(HERE, f"{prefix}_{args.source}_{args.train_end}.joblib")
    else:
        out = os.path.join(HERE, f"{prefix}.joblib")

    meta["feature_source"] = args.feature_source   # 记录特征源（跨标的模式用），供推理侧参考
    joblib.dump(meta, out)
    print(f"固定模型已保存: {out}")
    if args.train_end:
        print(
            "样本外测试命令（例如）：\n"
            f"  /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/run_any_backtest.py "
            f"--target {args.source} --model {os.path.basename(out)} "
            f"--test-start 2025-01-01"
        )


if __name__ == "__main__":
    main()
