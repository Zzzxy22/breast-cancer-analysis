"""端到端测试：完整跑一遍分析流程，检查产出文件与模型质量。"""

from bc_analysis.__main__ import run


def test_full_pipeline(tmp_path):
    report = run(csv=None, out=tmp_path, seed=0)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "核心结论" in text

    figs = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert figs == [
        "confusion_matrix.png",
        "correlation_heatmap.png",
        "effect_sizes.png",
        "logreg_coefficients.png",
        "roc_curves.png",
        "top_feature_distributions.png",
    ]
    for name in ("feature_tests.csv", "cv_scores.csv"):
        assert (tmp_path / name).exists()


def test_model_quality():
    from bc_analysis.data import load_dataset, split_xy
    from bc_analysis.model import evaluate

    X, y = split_xy(load_dataset())
    res = evaluate(X, y, random_state=1)
    assert res.test_metrics["roc_auc"] > 0.95
    assert res.confusion.sum() == res.test_metrics["n_test"]
