"""命令行入口：python -m bc_analysis"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import plots
from .data import load_dataset, split_xy
from .model import evaluate
from .report import build_report
from .stats import class_balance, compare_groups, correlated_pairs


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="乳腺肿瘤诊断数据分析：统计检验、可视化与建模")
    p.add_argument("--csv", type=Path, default=None, help="可选：同结构的 CSV 数据文件（默认使用内置数据集）")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="输出目录（默认 outputs/）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args(argv)


def run(csv: Path | None, out: Path, seed: int = 42) -> Path:
    fig_dir = out / "figures"
    print("[1/5] 加载并清洗数据")
    df = load_dataset(csv)

    print("[2/5] 统计检验")
    balance = class_balance(df)
    tests = compare_groups(df)
    pairs = correlated_pairs(df)
    out.mkdir(parents=True, exist_ok=True)
    tests.to_csv(out / "feature_tests.csv", index=False)

    print("[3/5] 训练与评估模型")
    X, y = split_xy(df)
    results = evaluate(X, y, random_state=seed)
    results.cv_scores.to_csv(out / "cv_scores.csv", index=False)

    print("[4/5] 生成图表")
    mean_feats = [c for c in df.columns if c.startswith("mean_")]
    figures = {
        "effect_sizes": plots.plot_effect_sizes(tests, fig_dir / "effect_sizes.png"),
        "distributions": plots.plot_top_distributions(
            df, tests["feature"].head(6).tolist(), fig_dir / "top_feature_distributions.png"
        ),
        "correlation": plots.plot_correlation(df, mean_feats, fig_dir / "correlation_heatmap.png"),
        "roc": plots.plot_roc(results.roc_curves, fig_dir / "roc_curves.png"),
        "confusion": plots.plot_confusion(
            results.confusion, results.best_name, fig_dir / "confusion_matrix.png"
        ),
        "coefficients": plots.plot_coefficients(
            results.coefficients, fig_dir / "logreg_coefficients.png"
        ),
    }

    print("[5/5] 写入报告")
    report = build_report(df, balance, tests, pairs, results, figures, out / "report.md")
    tm = results.test_metrics
    print(
        f"\n完成！最佳模型：{results.best_name} | 测试集 AUC {tm['roc_auc']:.3f} | "
        f"准确率 {tm['accuracy']:.1%} | 召回率 {tm['recall']:.1%}"
    )
    print(f"报告：{report}")
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args.csv, args.out, args.seed)


if __name__ == "__main__":
    main()
