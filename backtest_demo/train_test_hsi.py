"""HSI 训练 / 回测 严格分离（无未来函数）。

本脚本把“训练”和“回测”两个步骤明确拆开，并强制约束二者区间不重叠：

    阶段一  TRAIN   只取 train_end（含）及之前的数据训练模型（默认纯AI 2D CNN，
                     可由 --model-type rf 切回随机森林），冻结保存为固定模型。
                     该模型在阶段二运行期间绝不再重训。
    阶段二  TEST    用冻结模型对 test_start（含）及之后的数据做样本外推理；
                     test_start 之前的数据仅作预热（构造特征/价格图），不参与训练、也不交易。

核心不变量（任一违反即报错退出）：  train_end < test_start
    —— 保证训练期与测试期 100% 不重叠，彻底杜绝未来函数泄漏。

说明：
    - 训练与回测共用同一数据源（yfinance ^HSI，2016 起）。
    - CNN 模式（默认）：用最近 lookback 根收盘价铺成 L×L 价格图（相对价归一化），
      纯 AI、无任何手工技术指标（见 ai_strategy.cnn_image / make_cnn_dataset）。
    - RF 模式：用 13 维无量纲手工特征（见 ai_strategy.feature_at / make_dataset）。
    - run_any_backtest.py 的默认 MODEL_PATH 仍是 v0.0.5 时期的泄漏模型 rf_model.joblib
      （保留作历史存档，能跑出 +59.56%）。本脚本**显式指定**干净模型
      （默认 cnn_model_HSI_2021-12-31.joblib），因此默认给出的是诚实的样本外结果。

用法（仓库根目录执行，需先装好 venv：/Users/zhuxiaodong/.venvs/btvenv）：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/train_test_hsi.py

自定义训练/测试切分：
    /Users/zhuxiaodong/.venvs/btvenv/bin/python backtest_demo/train_test_hsi.py \
        --train-end 2021-12-31 --test-start 2022-01-01
"""
import argparse
import datetime
import os
import subprocess
import sys

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _run(cmd: list) -> None:
    """以仓库根目录 + PYTHONPATH 调用子脚本，实时透传输出。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    print("\n$ " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
    if rc != 0:
        raise SystemExit(f"子命令失败（exit={rc}）: {' '.join(cmd)}")


def phase_train(source: str, train_end: str, model_out: str, model_type: str = "rf",
                regression: bool = False) -> dict:
    _banner(f"阶段一  TRAIN  ▶ 只训练 {source} 在 {train_end}（含）之前的数据"
            + ("（纯AI CNN" + ("·回归(归一化5日涨幅)" if regression else "·分类(涨跌)") + "）"
               if model_type == "cnn" else "（随机森林）"))
    out_path = model_out if os.path.isabs(model_out) else os.path.join(HERE, model_out)
    cmd = [
        sys.executable, os.path.join(HERE, "train_model.py"),
        "--source", source, "--train-end", train_end, "--out", out_path,
        "--model-type", model_type,
    ]
    if model_type == "cnn":
        cmd += ["--epochs", "60"]
        if regression:
            cmd += ["--regression"]
    _run(cmd)
    blob = joblib.load(out_path)
    meta = {k: v for k, v in blob.items() if k != "model"}
    print("\n[模型元数据]")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return meta


def phase_test(target: str, model_out: str, test_start: str, no_kelly: bool = False) -> None:
    _banner(f"阶段二  TEST   ▶ 用冻结模型对 {target} 自 {test_start} 起做样本外推理"
            + ("（二值满仓/空仓，无凯利）" if no_kelly else "（凯利百分比仓位）"))
    cmd = [
        sys.executable, os.path.join(HERE, "run_any_backtest.py"),
        "--target", target, "--model", model_out, "--test-start", test_start,
    ]
    if no_kelly:
        cmd.append("--no-kelly")
    _run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="HSI 训练/回测严格分离（无未来函数）")
    ap.add_argument("--source", default="HSI", help="训练源标的（默认 HSI）")
    ap.add_argument("--target", default="HSI", help="回测标的（默认 HSI）")
    ap.add_argument("--train-end", default="2021-12-31",
                    help="训练数据截止日(含)，YYYY-MM-DD。默认 2021-12-31")
    ap.add_argument("--test-start", default="2022-01-01",
                    help="样本外测试起点(含)，YYYY-MM-DD。默认 2022-01-01")
    ap.add_argument("--model-out", default=None,
                    help="干净模型输出路径；默认 <rf|cnn>_model_<source>_<train_end>.joblib")
    ap.add_argument("--model-type", default="cnn", choices=["rf", "cnn"],
                    help="模型类型（默认 cnn=纯AI 2D CNN；rf=随机森林）。")
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接复用已存在的干净模型做测试")
    ap.add_argument("--no-kelly", action="store_true",
                    help="去掉凯利百分比仓位，改用二值仓位(满仓100%/空仓0%)")
    ap.add_argument("--no-regression", action="store_true",
                    help="CNN 退化为分类(涨跌)模式（默认 CNN 即回归：归一化5日涨幅→凯利仓位）")
    args = ap.parse_args()

    # 默认 CNN 即回归模式（归一化5日涨幅→凯利仓位）；rf 恒为分类。--no-regression 可关。
    regression = (args.model_type == "cnn") and (not args.no_regression)

    train_end_dt = datetime.datetime.strptime(args.train_end, "%Y-%m-%d").date()
    test_start_dt = datetime.datetime.strptime(args.test_start, "%Y-%m-%d").date()

    # —— 核心不变量：训练期与测试期不得重叠 ——
    if not (train_end_dt < test_start_dt):
        raise SystemExit(
            f"❌ 训练/测试区间重叠（未来函数泄漏风险）！\n"
            f"   要求 train_end({args.train_end}) < test_start({args.test_start})，"
            f"请增大间隔（如 train_end=2021-12-31, test-start=2022-01-01）。"
        )

    if args.model_type == "cnn":
        prefix = "cnn_ret_model" if regression else "cnn_model"
    else:
        prefix = "rf_model"
    model_out = args.model_out or f"{prefix}_{args.source}_{args.train_end}.joblib"
    out_path = model_out if os.path.isabs(model_out) else os.path.join(HERE, model_out)

    _banner("HSI 训练 / 回测 严格分离")
    print(f"训练区间 : 2016-01-01 ~ {args.train_end}  （{args.source} 全量截断到该日）")
    print(f"测试区间 : {args.test_start} ~ 今（test_start 之前仅预热、不交易）")
    print(f"干净模型 : {model_out}")
    print(f"重叠检查 : train_end({args.train_end}) < test_start({args.test_start}) ✅ 通过")

    if args.no_train and os.path.exists(out_path):
        _banner(f"阶段一  跳过（--no-train，复用现有 {model_out}）")
        blob = joblib.load(out_path)
        meta = {k: v for k, v in blob.items() if k != "model"}
        print("[已存在模型元数据]")
        for k, v in meta.items():
            print(f"  {k}: {v}")
    else:
        meta = phase_train(args.source, args.train_end, model_out,
                           model_type=args.model_type, regression=regression)
        # 二次校验：模型自身记录的 train_end 必须早于测试起点
        mte = meta.get("train_end")
        if mte:
            mte_dt = datetime.datetime.strptime(mte, "%Y-%m-%d").date()
            if not (mte_dt < test_start_dt):
                raise SystemExit(
                    f"❌ 模型内 train_end={mte} 不早于 test_start={args.test_start}，"
                    f"存在未来函数泄漏，已中止。"
                )
            print(f"\n[校验] 模型 train_end={mte} < test_start={args.test_start} ✅")

    phase_test(args.target, model_out, args.test_start, no_kelly=args.no_kelly)

    _banner("完成：训练与回测已分离，测试期为严格样本外（无未来函数）")
    print(f"  训练模型 : {model_out}  (train_end={meta.get('train_end')})")
    print(f"  测试起点 : {args.test_start}  （该日之前仅预热，不参与训练/交易）")
    print(f"  结果文件 : backtest_demo/frozen_HSI_daily_result.csv")
    print(f"             backtest_demo/frozen_HSI_chart.html")


if __name__ == "__main__":
    main()
